"""Tests for persist_generated_name_batch stage-transition wiring.

Verifies that :func:`~imas_codex.standard_names.graph_ops.persist_generated_name_batch`
correctly transitions StandardName stage fields, clears source claims, and
creates PRODUCED_NAME edges — all in a single Neo4j transaction.

Tests mock :class:`GraphClient` — no live Neo4j required.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — mirrors the pattern from test_seed_expand_claims.py
# ---------------------------------------------------------------------------


def _mock_gc_tx():
    """Build a mock GraphClient with transaction support.

    Returns ``(gc, tx)`` where *gc* is the mock GraphClient and *tx* is the
    mock Transaction.
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc, tx


def _mock_gc_query():
    """Build a mock GraphClient that supports ``gc.query()`` only."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    # write_standard_names needs unit conflict check + the write query
    gc.query = MagicMock(return_value=[{"count": 1}])
    return gc


def _patch_gc(mock_gc):
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


def _make_candidate(
    *,
    name: str = "electron_temperature",
    source_id: str = "dd:core_profiles/profiles_1d/electrons/temperature",
    model: str = "test/model",
) -> dict:
    return {
        "id": name,
        "source_id": source_id,
        "source_types": ["dd"],
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": ["core_profiles"],
        "model": model,
        "llm_model": model,
        "llm_service": "standard-names",
    }


# ---------------------------------------------------------------------------
# Unit tests for _finalize_generated_name_stage
# ---------------------------------------------------------------------------


class TestFinalizeGeneratedNameStage:
    """Tests for the atomic finalize helper directly."""

    def test_single_transaction_committed(self):
        """The finalize step commits in a single transaction."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [
                    {
                        "sn_id": "electron_temperature",
                        "sns_id": "dd:core_profiles/profiles_1d/electrons/temperature",
                        "model": "test/model",
                    }
                ]
            )

        tx.run.assert_called_once()
        tx.commit.assert_called_once()

    def test_sets_name_stage_drafted(self):
        """The Cypher query sets name_stage = 'drafted'."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "e_temp", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher = tx.run.call_args.args[0]
        assert "name_stage" in cypher
        assert "'drafted'" in cypher

    def test_sets_chain_length_zero(self):
        """The Cypher query sets chain_length = 0."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "e_temp", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher = tx.run.call_args.args[0]
        assert "chain_length" in cypher
        assert "= 0" in cypher

    def test_sets_docs_stage_pending(self):
        """The Cypher query sets docs_stage = 'pending'."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "e_temp", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher = tx.run.call_args.args[0]
        assert "docs_stage" in cypher
        assert "'pending'" in cypher

    def test_clears_claim_on_source(self):
        """The Cypher query clears claim_token and claimed_at on the source."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "e_temp", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher = tx.run.call_args.args[0]
        assert "claim_token" in cypher
        assert "null" in cypher
        assert "claimed_at" in cypher

    def test_creates_produced_name_edge(self):
        """The Cypher query creates a PRODUCED_NAME edge."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "e_temp", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher = tx.run.call_args.args[0]
        assert "PRODUCED_NAME" in cypher
        assert "MERGE" in cypher

    def test_sets_source_status_composed(self):
        """The Cypher query sets the source status to 'composed'."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [{"sn_id": "e_temp", "sns_id": "dd:p/q", "model": "m"}]
            )

        cypher = tx.run.call_args.args[0]
        assert "status" in cypher
        assert "'composed'" in cypher

    def test_rollback_on_exception(self):
        """If the transaction run raises, close() is called and exception re-raised."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        tx.run.side_effect = RuntimeError("neo4j failure")

        with _patch_gc(gc):
            with pytest.raises(RuntimeError, match="neo4j failure"):
                _finalize_generated_name_stage(
                    [{"sn_id": "e_temp", "sns_id": "dd:p/q", "model": "m"}]
                )

        tx.commit.assert_not_called()
        tx.close.assert_called_once()

    def test_empty_batch_is_noop(self):
        """An empty batch calls no graph operations."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage([])

        tx.run.assert_not_called()
        tx.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests for persist_generated_name_batch
# ---------------------------------------------------------------------------


class TestPersistGeneratedNameBatch:
    """End-to-end tests for persist_generated_name_batch."""

    def _make_persist_patches(self, gc_query, gc_tx):
        """Return combined patches: query client for write_standard_names,
        tx client for _finalize_generated_name_stage.

        We need a side_effect on GraphClient that returns gc_query on the
        first call (write_standard_names) and gc_tx on the second call
        (_finalize_generated_name_stage).
        """
        call_count = {"n": 0}

        class _SwitchingGC:
            def __init__(self):
                self._idx = call_count["n"]
                call_count["n"] += 1
                self._gc = gc_query if self._idx == 0 else gc_tx

            def __enter__(self):
                return self._gc.__enter__()

            def __exit__(self, *args):
                return self._gc.__exit__(*args)

        return patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            side_effect=_SwitchingGC,
        )

    def test_persist_calls_finalize_stage(self):
        """persist_generated_name_batch calls _finalize_generated_name_stage."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidates = [_make_candidate()]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops._finalize_generated_name_stage"
            ) as mock_finalize,
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                return_value=0,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        mock_finalize.assert_called_once()
        finalize_batch = mock_finalize.call_args.args[0]
        assert len(finalize_batch) == 1
        assert finalize_batch[0]["sn_id"] == "electron_temperature"
        assert finalize_batch[0]["model"] == "test/model"

    def test_persist_sets_name_stage_drafted(self):
        """persist_generated_name_batch triggers name_stage='drafted' on the SN."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc_tx, tx = _mock_gc_tx()
        candidates = [_make_candidate()]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc_tx,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        tx.run.assert_called_once()
        cypher = tx.run.call_args.args[0]
        assert "name_stage" in cypher and "'drafted'" in cypher

    def test_persist_sets_chain_length_zero(self):
        """persist_generated_name_batch sets chain_length=0 on new SN."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc_tx, tx = _mock_gc_tx()
        candidates = [_make_candidate()]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc_tx,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        cypher = tx.run.call_args.args[0]
        assert "chain_length" in cypher and "= 0" in cypher

    def test_persist_sets_docs_stage_pending(self):
        """persist_generated_name_batch sets docs_stage='pending' on new SN."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc_tx, tx = _mock_gc_tx()
        candidates = [_make_candidate()]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc_tx,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        cypher = tx.run.call_args.args[0]
        assert "docs_stage" in cypher and "'pending'" in cypher

    def test_persist_clears_claim_on_source(self):
        """persist_generated_name_batch clears claim_token/claimed_at on source."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc_tx, tx = _mock_gc_tx()
        candidates = [_make_candidate()]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc_tx,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        cypher = tx.run.call_args.args[0]
        assert "claim_token" in cypher
        assert "null" in cypher

    def test_persist_creates_produced_name_edge(self):
        """persist_generated_name_batch creates PRODUCED_NAME edge."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc_tx, tx = _mock_gc_tx()
        candidates = [_make_candidate()]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc_tx,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        cypher = tx.run.call_args.args[0]
        assert "PRODUCED_NAME" in cypher

    def test_persist_idempotent_merge_semantics(self):
        """persist_generated_name_batch uses MERGE so re-running is idempotent."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc_tx, tx = _mock_gc_tx()
        candidates = [_make_candidate()]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc_tx,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            # First run
            persist_generated_name_batch(candidates, compose_model="test/model")
            # Second run (idempotent — MERGE on SNS→SN edge means no duplicate)
            persist_generated_name_batch(candidates, compose_model="test/model")

        # Both calls should commit exactly once each (two calls total)
        assert tx.commit.call_count == 2

    def test_persist_empty_candidates_returns_zero(self):
        """Empty candidate list returns 0 without touching the graph."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc_tx, tx = _mock_gc_tx()

        with _patch_gc(gc_tx):
            result = persist_generated_name_batch([], compose_model="test/model")

        assert result == 0
        tx.run.assert_not_called()

    def test_persist_error_sibling_excluded_from_finalize(self):
        """Error-sibling candidates (no source node) are excluded from finalize."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        error_sibling = _make_candidate(name="error_sibling", source_id="dd:x/y")
        error_sibling["model"] = "deterministic:dd_error_modifier"
        normal_cand = _make_candidate()
        candidates = [error_sibling, normal_cand]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=2,
            ),
            patch(
                "imas_codex.standard_names.graph_ops._finalize_generated_name_stage"
            ) as mock_finalize,
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                return_value=0,
            ) as mock_supersede,
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        finalize_batch = mock_finalize.call_args.args[0]
        ids = [item["sn_id"] for item in finalize_batch]
        assert "error_sibling" not in ids
        assert "electron_temperature" in ids

        # The deterministic error-sibling must NOT be a supersede candidate
        # (it owns a distinct error-node source); only the LLM-composed name
        # participates in the one-source-one-name invariant.
        supersede_pairs = mock_supersede.call_args.args[0]
        pair_names = {p["new_name"] for p in supersede_pairs}
        assert "error_sibling" not in pair_names
        assert "electron_temperature" in pair_names


# ---------------------------------------------------------------------------
# One-source-one-name invariant (Class-A: forced regen supersedes prior name)
# ---------------------------------------------------------------------------


class TestSupersedePriorSourceNames:
    """``supersede_prior_source_names`` retires stale pipeline names left on a
    source by a ``--force``/regen pass, enforcing the invariant
    *one source → at most one non-superseded pipeline name*.
    """

    def test_empty_pairs_is_noop(self):
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        # No GraphClient is opened when there is nothing to do.
        assert supersede_prior_source_names([]) == 0

    def test_pairs_missing_fields_filtered_before_graph(self):
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        # Pairs lacking new_name or source_id are dropped before any
        # GraphClient connection (so this stays a default-tier test).
        result = supersede_prior_source_names(
            [{"new_name": "", "source_id": "dd:x"}, {"new_name": "n", "source_id": ""}]
        )
        assert result == 0

    def test_supersedes_prior_pipeline_name(self):
        """A regen that produced a *different* name supersedes the prior
        accepted pipeline name on the same source — leaving one live name."""
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        gc = _mock_gc_query()
        # The Cypher returns the (old, new) row for each predecessor it retired.
        gc.query = MagicMock(
            return_value=[
                {"old_name": "old_pipeline_name", "new_name": "new_pipeline_name"}
            ]
        )

        with _patch_gc(gc):
            n = supersede_prior_source_names(
                [{"new_name": "new_pipeline_name", "source_id": "dd:eq/q_95"}]
            )

        assert n == 1
        cypher = gc.query.call_args.args[0]
        # The predecessor is marked superseded and linked via REFINED_FROM.
        assert "old.name_stage = 'superseded'" in cypher
        assert "MERGE (new)-[:REFINED_FROM]->(old)" in cypher
        # Only pipeline-origin predecessors are touched — catalog_edit and
        # derived names are excluded by the WHERE clause.
        assert "coalesce(old.origin, 'pipeline') = 'pipeline'" in cypher
        # Already-retired names are never re-superseded.
        assert "['superseded', 'exhausted']" in cypher
        # The new name itself is never superseded (byte-identical regen no-op).
        assert "old.id <> pr.new_name" in cypher

    def test_open_edit_propagated_to_successor_and_predecessor_reconciled(self):
        """A name-hint regen that supersedes the edited predecessor must ride
        the still-open edit forward onto the recomposed successor and reconcile
        the predecessor to 'applied' — otherwise the edit is stuck 'open' on a
        superseded node forever."""
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        gc = _mock_gc_query()
        gc.query = MagicMock(
            return_value=[{"old_name": "old_name", "new_name": "new_name"}]
        )

        with _patch_gc(gc):
            supersede_prior_source_names(
                [{"new_name": "new_name", "source_id": "dd:eq/q_95"}]
            )

        cypher = gc.query.call_args.args[0]
        # The propagation is gated on the predecessor's edit still being open.
        assert "(coalesce(old.edit_status, '') = 'open') AS carry_edit" in cypher
        # Predecessor reconciled to 'applied' (no longer stuck 'open').
        assert "old.edit_status = CASE WHEN carry_edit THEN 'applied'" in cypher
        # Successor inherits the open-edit steering fields …
        assert "new.name_hint = CASE WHEN carry_edit" in cypher
        assert "new.edit_reason = CASE WHEN carry_edit" in cypher
        assert "new.edit_scope = CASE WHEN carry_edit" in cypher
        # … including the cascade-authorization opt-in flags (item-1 flags) …
        assert "new.edit_override_edits = CASE WHEN carry_edit" in cypher
        assert "new.edit_include_accepted = CASE WHEN carry_edit" in cypher
        # … and the open status itself so it resolves at review time.
        assert "new.edit_status = CASE WHEN carry_edit" in cypher

    def test_byte_identical_regen_is_noop(self):
        """When the regenerated name equals the existing name (same node id),
        the WHERE ``old.id <> pr.new_name`` clause excludes it — nothing is
        superseded."""
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[])  # no predecessor distinct from new

        with _patch_gc(gc):
            n = supersede_prior_source_names(
                [{"new_name": "same_name", "source_id": "dd:eq/q_95"}]
            )

        assert n == 0

    def test_persist_regen_supersedes_one_live_name(self):
        """End-to-end: a forced regen producing a different name for an
        already-named source ends with exactly one live (non-superseded)
        pipeline name for that source.

        Models the graph with an in-memory store driven through the real
        ``supersede_prior_source_names`` Cypher contract (mocked execution).
        """
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        # Track which names the persist path asked to supersede.
        captured: dict[str, object] = {}

        def _fake_supersede(pairs):
            captured["pairs"] = pairs
            # Simulate: the source already had 'old_name'; it is now retired,
            # leaving only the freshly-composed name live.
            return len(pairs)

        candidates = [
            _make_candidate(
                name="safety_factor_of_flux_surface",
                source_id="equilibrium/time_slice/global_quantities/q_95",
            )
        ]

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage"),
            patch("imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"),
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                side_effect=_fake_supersede,
            ),
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=None,
            ),
        ):
            persist_generated_name_batch(candidates, compose_model="test/model")

        # The persist path must have routed the new name + its DD source to the
        # supersede invariant guard.
        pairs = captured["pairs"]
        assert pairs == [
            {
                "new_name": "safety_factor_of_flux_surface",
                "source_id": "equilibrium/time_slice/global_quantities/q_95",
            }
        ]

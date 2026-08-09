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

    def _run(_cypher, **params):
        if "GENERATED_SUPERSESSION_PREFLIGHT" in _cypher:
            return [
                {
                    "requested_source_id": pair["source_id"],
                    "new_name": pair["new_name"],
                    "requested_source_exists": True,
                    "successor_exists": True,
                    "trigger_source_id": f"dd:{pair['source_id']}",
                    "old_name": None,
                    "old_stage": None,
                    "judged_source_ids": [],
                    "retained_source_ids": [],
                }
                for pair in params["pairs"]
            ]
        return [
            {"id": item["sns_id"]}
            for item in params.get("batch", [])
            if item.get("sns_id")
        ]

    tx = MagicMock()
    tx.closed = False
    tx.run = MagicMock(side_effect=_run)
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


def _transaction_call(tx, fragment: str):
    """Return the first transaction call whose Cypher contains *fragment*."""
    return next(
        call_item
        for call_item in tx.run.call_args_list
        if fragment in call_item.args[0]
    )


def _make_candidate(
    *,
    name: str = "electron_temperature",
    source_id: str = "core_profiles/profiles_1d/electrons/temperature",
    model: str = "test/model",
) -> dict:
    return {
        "id": name,
        "source_id": source_id,
        "source_types": ["dd"],
        "kind": "scalar",
        "description": "Electron temperature in the plasma core",
        "unit": "eV",
        "physics_domain": ["core_profiles"],
        "model": model,
        "llm_model": model,
        "llm_service": "standard-names",
        "source_claim_token": "winner",
        "source_claim_seq": 7,
    }


def test_descriptionless_candidate_releases_source_without_persistence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A legacy candidate without prose is released for a clean retry."""
    from imas_codex.standard_names.workers import (
        _persist_description_checked_candidates,
    )

    candidate = _make_candidate()
    candidate["description"] = " \t"

    with (
        patch(
            "imas_codex.standard_names.graph_ops.release_generate_name_failed_claims",
            return_value=1,
        ) as release,
        patch(
            "imas_codex.standard_names.graph_ops.persist_generated_name_batch",
        ) as persist,
        caplog.at_level("WARNING"),
    ):
        result = _persist_description_checked_candidates(
            [candidate],
            source_type="dd",
            phase="generate_name",
            compose_model="test/model",
            dd_version="4.1.0",
            cocos_version=None,
            run_id="run-1",
        )

    assert result == []
    release.assert_called_once_with(
        source_ids=[
            "dd:core_profiles/profiles_1d/electrons/temperature",
        ],
        claim_token="winner",
    )
    persist.assert_not_called()
    assert "rejected 1 candidate(s) with empty descriptions" in caplog.text
    assert "released 1/1 exact source claim(s) for retry" in caplog.text


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

    def test_existing_accepted_name_is_not_demoted(self):
        """Finalize only initializes pending state and preserves later lifecycle."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        with _patch_gc(gc):
            _finalize_generated_name_stage(
                [
                    {
                        "sn_id": "electron_temperature",
                        "sns_id": "dd:core_profiles/temperature",
                        "model": "test/model",
                        "claim_token": "winner",
                        "claim_seq": 9,
                    }
                ]
            )

        cypher = tx.run.call_args.args[0]
        assert "ELSE sn.name_stage" in cypher
        assert "sn.docs_stage = coalesce(sn.docs_stage, 'pending')" in cypher
        assert "sns.claim_token = b.claim_token" in cypher
        assert "sns.claim_seq = b.claim_seq" in cypher
        assert "sns.last_error   = null" in cypher
        assert "source_gap:HAS_STANDARD_NAME_VOCAB_GAP" in cypher
        assert "entity_gap:HAS_STANDARD_NAME_VOCAB_GAP" in cypher
        assert "size(backing_entities) = 1" in cypher
        assert "sns.id = 'dd:' + sns.source_id" in cypher
        assert "DELETE edge" in cypher
        assert "DELETE vg" not in cypher
        assert "HAS_EVIDENCE" not in cypher

    def test_stale_source_fence_precedes_gap_retirement(self):
        """A lost source claim cannot reach relationship retirement."""
        from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

        gc, tx = _mock_gc_tx()
        tx.run.side_effect = lambda *_args, **_kwargs: []
        with _patch_gc(gc):
            assert (
                _finalize_generated_name_stage(
                    [
                        {
                            "sn_id": "electron_temperature",
                            "sns_id": "dd:core_profiles/temperature",
                            "model": "test/model",
                            "claim_token": "stale",
                            "claim_seq": 9,
                        }
                    ]
                )
                == []
            )

        cypher = tx.run.call_args.args[0]
        assert cypher.index("sns.claim_token = b.claim_token") < cypher.index(
            "source_gap:HAS_STANDARD_NAME_VOCAB_GAP"
        )
        assert cypher.index("sns.claim_seq = b.claim_seq") < cypher.index(
            "entity_gap:HAS_STANDARD_NAME_VOCAB_GAP"
        )

    def test_finalize_batch_carries_exact_source_fence(self):
        """Persistence threads the source token and sequence through one tx."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        gc, tx = _mock_gc_tx()
        candidate = _make_candidate()
        candidate["source_claim_token"] = "winner"
        candidate["source_claim_seq"] = 12

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                return_value=0,
            ),
        ):
            persist_generated_name_batch([candidate], compose_model="test/model")

        item = tx.run.call_args_list[0].kwargs["batch"][0]
        assert item["claim_token"] == "winner"
        assert item["claim_seq"] == 12

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
        assert "coalesce(sn.chain_length, 0)" in cypher

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


def test_candidate_stage_reserves_missing_target_under_source_fence() -> None:
    from imas_codex.standard_names.graph_ops import (
        stage_claimed_generated_candidates,
    )

    gc, tx = _mock_gc_tx()
    with _patch_gc(gc):
        winners = stage_claimed_generated_candidates(
            [
                {
                    "sns_id": "dd:equilibrium/time_slice/q",
                    "source_id": "equilibrium/time_slice/q",
                    "sn_id": "safety_factor",
                    "unit": "1",
                    "model": "test/model",
                    "claim_token": "new-winner",
                    "claim_seq": 9,
                }
            ]
        )

    assert winners == ["dd:equilibrium/time_slice/q"]
    lock_cypher = tx.run.call_args_list[0].args[0]
    assert "sns.claim_token = b.claim_token" in lock_cypher
    assert "sns.claim_seq = b.claim_seq" in lock_cypher
    assert "sns.status = 'extracted'" in lock_cypher
    assert "MERGE (target:StandardName {id: b.sn_id})" in lock_cypher
    assert "ON CREATE SET target.created_at = datetime()" in lock_cypher
    assert "target.name_stage = $pending_stage" in lock_cypher
    assert "WITH b, sns, target\n        FOREACH" in lock_cypher
    assert "SET target.binding_lock_token = b.claim_token" in lock_cypher
    assert "REMOVE target.binding_lock_token, target.binding_lock_seq" in lock_cypher
    assert "MERGE (sns)-[reservation:PRODUCED_NAME]->(target)" in lock_cypher
    assert "reservation.provisional = true" in lock_cypher
    assert "reservation.claim_token = b.claim_token" in lock_cypher
    assert "reservation.claim_seq = b.claim_seq" in lock_cypher
    assert "reservation.created_target = true" in lock_cypher
    assert "HAS_STANDARD_NAME" not in lock_cypher
    cleanup_cypher = tx.run.call_args_list[1].args[0]
    assert "b.preserve_current = true" in cleanup_cypher
    assert tx.run.call_count == 2
    tx.commit.assert_called_once()


class TestClaimedAttachmentPersistence:
    """Claimed attachments share the stable-target lifecycle fence."""

    @staticmethod
    def _attachment() -> dict:
        return {
            "sns_id": "dd:spectrometer/channel/isotope_ratio",
            "source_id": "spectrometer/channel/isotope_ratio",
            "standard_name": "hydrogen_fraction",
            "claim_token": "attachment-winner",
            "claim_seq": 5,
        }

    def test_terminal_target_releases_without_attachment_mutation(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_claimed_attachments

        gc, tx = _mock_gc_tx()

        def _collision(cypher, **_params):
            if "AS outcome" not in cypher:
                return []
            return [
                {
                    "id": "dd:spectrometer/channel/isotope_ratio",
                    "outcome": "lifecycle_collision",
                    "candidate_id": "hydrogen_fraction",
                    "target_stage": "superseded",
                    "attempt_count": 4,
                }
            ]

        tx.run.side_effect = _collision
        with _patch_gc(gc):
            winners = persist_claimed_attachments([self._attachment()])

        assert winners == []
        assert tx.run.call_count == 3
        cypher = tx.run.call_args_list[0].args[0]
        assert "sns.claim_token = b.claim_token" in cypher
        assert "sns.claim_seq = b.claim_seq" in cypher
        release = tx.run.call_args_list[2].args[0]
        assert "sns.last_error = CASE" in release
        assert "MERGE (source)-[:HAS_STANDARD_NAME]" not in release
        tx.commit.assert_called_once()

    def test_accepted_target_attaches_under_same_transaction(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_claimed_attachments

        gc, tx = _mock_gc_tx()

        def _winner(cypher, **params):
            batch = params.get("batch", [])
            if "AS outcome" in cypher:
                return [
                    {
                        "id": batch[0]["sns_id"],
                        "outcome": "winner",
                        "candidate_id": batch[0]["sn_id"],
                        "target_stage": "accepted",
                        "attempt_count": 1,
                    }
                ]
            return [{"id": item["sns_id"]} for item in batch]

        tx.run.side_effect = _winner
        with _patch_gc(gc):
            winners = persist_claimed_attachments([self._attachment()])

        assert winners == ["dd:spectrometer/channel/isotope_ratio"]
        cleanup = tx.run.call_args_list[1].args[0]
        mutation = tx.run.call_args_list[2].args[0]
        assert "b.preserve_current = true" in cleanup
        assert "WHERE sn.name_stage IN $stable_stages" in mutation
        assert "MERGE (sns)-[produced:PRODUCED_NAME]->(sn)" in mutation
        assert "REMOVE produced.provisional" in mutation
        assert "produced.claim_token" in mutation
        assert "produced.claim_seq" in mutation
        assert "sns.last_error = null" in mutation
        assert "MERGE (src)-[:HAS_STANDARD_NAME]->(sn)" in mutation
        tx.commit.assert_called_once()

    def test_missing_fence_never_opens_graph(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_claimed_attachments

        attachment = self._attachment()
        attachment["claim_token"] = None
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as graph:
            assert persist_claimed_attachments([attachment]) == []
        graph.assert_not_called()

    def test_final_write_failure_rolls_back_lifecycle_fence(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_claimed_attachments

        gc, tx = _mock_gc_tx()

        def _fail_after_lock(cypher, **params):
            if "AS outcome" in cypher:
                batch = params["batch"]
                return [
                    {
                        "id": batch[0]["sns_id"],
                        "outcome": "winner",
                        "candidate_id": batch[0]["sn_id"],
                        "target_stage": "accepted",
                        "attempt_count": 1,
                    }
                ]
            if "[stale:PRODUCED_NAME]" in cypher:
                return []
            raise RuntimeError("attachment write failed")

        tx.run.side_effect = _fail_after_lock
        with (
            _patch_gc(gc),
            pytest.raises(RuntimeError, match="attachment write failed"),
        ):
            persist_claimed_attachments([self._attachment()])

        tx.commit.assert_not_called()
        tx.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests for persist_generated_name_batch
# ---------------------------------------------------------------------------


class TestPersistGeneratedNameBatch:
    """End-to-end tests for persist_generated_name_batch."""

    @pytest.fixture(autouse=True)
    def _stage_exact_claims(self):
        gc, tx = _mock_gc_tx()
        self.atomic_tx = tx
        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=gc,
        ):
            yield

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
                "imas_codex.standard_names.graph_ops._finalize_generated_name_stage",
                return_value=["dd:core_profiles/profiles_1d/electrons/temperature"],
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

        mock_finalize.assert_not_called()
        finalize_batch = _transaction_call(
            self.atomic_tx, "sns.status = 'composed'"
        ).kwargs["batch"]
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

        cypher = _transaction_call(tx, "sns.status = 'composed'").args[0]
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

        cypher = _transaction_call(tx, "sns.status = 'composed'").args[0]
        assert "coalesce(sn.chain_length, 0)" in cypher

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

        cypher = _transaction_call(tx, "sns.status = 'composed'").args[0]
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

        cypher = _transaction_call(tx, "sns.status = 'composed'").args[0]
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

        cypher = _transaction_call(tx, "sns.status = 'composed'").args[0]
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

    def test_missing_source_fence_has_no_graph_side_effects(self):
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate()
        candidate["source_claim_token"] = None
        with (
            patch("imas_codex.standard_names.graph_ops.write_standard_names") as write,
            patch(
                "imas_codex.standard_names.graph_ops._finalize_generated_name_stage"
            ) as finalize,
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"
            ) as backfill,
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names"
            ) as supersede,
            patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter") as counter,
        ):
            result = persist_generated_name_batch(
                [candidate], compose_model="test/model"
            )

        assert result == 0
        write.assert_not_called()
        finalize.assert_not_called()
        backfill.assert_not_called()
        supersede.assert_not_called()
        counter.assert_not_called()

    def test_claim_turnover_before_lock_has_no_rich_write(self):
        """A claim lost before the tx lock cannot write a candidate."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        self.atomic_tx.run.side_effect = lambda _cypher, **_params: []
        with patch("imas_codex.standard_names.graph_ops.write_standard_names") as write:
            result = persist_generated_name_batch(
                [_make_candidate()], compose_model="test/model"
            )

        assert result == 0
        write.assert_not_called()
        self.atomic_tx.commit.assert_called_once()

    def test_terminal_same_id_releases_source_without_rich_mutation(self):
        """A terminal same-id node is a diagnosed retry, never a success."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate(name="hydrogen_fraction")

        def _terminal_collision(cypher, **params):
            assert params["batch"][0]["claim_token"] == "winner"
            assert params["batch"][0]["claim_seq"] == 7
            if "AS outcome" not in cypher:
                return []
            return [
                {
                    "id": "dd:core_profiles/profiles_1d/electrons/temperature",
                    "outcome": "lifecycle_collision",
                    "candidate_id": "hydrogen_fraction",
                    "target_stage": "superseded",
                    "attempt_count": 3,
                }
            ]

        self.atomic_tx.run.side_effect = _terminal_collision
        with (
            patch("imas_codex.standard_names.graph_ops.write_standard_names") as write,
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"
            ) as backfill,
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names"
            ) as supersede,
            patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter") as counter,
        ):
            winners = persist_generated_name_batch(
                [candidate],
                compose_model="test/model",
                return_winner_ids=True,
            )

        assert winners == []
        write.assert_not_called()
        backfill.assert_not_called()
        supersede.assert_not_called()
        counter.assert_not_called()
        self.atomic_tx.commit.assert_called_once()
        assert self.atomic_tx.run.call_count == 3
        cypher = self.atomic_tx.run.call_args_list[0].args[0]
        cleanup_cypher = self.atomic_tx.run.call_args_list[1].args[0]
        release_cypher = self.atomic_tx.run.call_args_list[2].args[0]
        assert "SET sns.claimed_at = datetime()" in cypher
        assert "stale.provisional = true" in cleanup_cypher
        assert "sns.last_error = CASE" in release_cypher
        assert 'candidate "' in release_cypher
        assert "target_stage" in cypher
        assert "attempt_count" in cypher
        assert "HAS_STANDARD_NAME" not in cypher

    @pytest.mark.parametrize("target_stage", ["accepted", "approved"])
    def test_existing_stable_same_id_is_provenance_only(self, target_stage):
        """Catalog-stable same-id reuse cannot rewrite target content."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate()

        def _accepted_winner(cypher, **params):
            batch = params.get("batch", [])
            if "AS outcome" in cypher:
                return [
                    {
                        "id": batch[0]["sns_id"],
                        "outcome": "winner",
                        "candidate_id": batch[0]["sn_id"],
                        "target_stage": target_stage,
                        "attempt_count": 2,
                    }
                ]
            return [{"id": item["sns_id"]} for item in batch]

        self.atomic_tx.run.side_effect = _accepted_winner
        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=[candidate["id"]],
            ) as write,
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"
            ) as backfill,
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                return_value=0,
            ) as supersede,
        ):
            winners = persist_generated_name_batch(
                [candidate],
                compose_model="test/model",
                return_winner_ids=True,
            )

        assert winners == ["dd:core_profiles/profiles_1d/electrons/temperature"]
        write.assert_not_called()
        backfill.assert_not_called()
        supersede.assert_not_called()
        lock_cypher = self.atomic_tx.run.call_args_list[0].args[0]
        assert "SET target.binding_lock_token = b.claim_token" in lock_cypher
        assert "target.binding_lock_seq = b.claim_seq" in lock_cypher
        assert (
            "REMOVE target.binding_lock_token, target.binding_lock_seq" in lock_cypher
        )
        cleanup_cypher = self.atomic_tx.run.call_args_list[1].args[0]
        assert "b.preserve_current = true" in cleanup_cypher
        finalize_cypher = _transaction_call(
            self.atomic_tx, "sns.status = 'composed'"
        ).args[0]
        assert "WHERE sn.name_stage IN $stable_stages" in finalize_cypher
        assert "MERGE (source)-[:HAS_STANDARD_NAME]->(sn)" in finalize_cypher
        assert "SET sn.source_paths = CASE" in finalize_cypher
        immutable_fields = (
            "description",
            "documentation",
            "source_types",
            "validation_status",
            "model",
            "grammar_parse_version",
            "review_input_hash",
            "name_stage",
            "docs_stage",
            "unit",
        )
        for field in immutable_fields:
            assert not any(
                line.lstrip().startswith(f"sn.{field} =")
                for line in finalize_cypher.splitlines()
            )

    @pytest.mark.parametrize(
        ("binding_kind", "target_stage", "rich_write_expected"),
        [
            ("owned_reservation", "pending", True),
            ("stable_reuse", "accepted", False),
        ],
    )
    def test_winner_prunes_stale_different_candidate_before_finalize(
        self,
        binding_kind,
        target_stage,
        rich_write_expected,
    ):
        """Both winner modes prune old provisional state before composing once."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate(name="electron_temperature")

        def _winner(cypher, **params):
            if "AS outcome" in cypher:
                item = params["batch"][0]
                return [
                    {
                        "id": item["sns_id"],
                        "outcome": "winner",
                        "binding_kind": binding_kind,
                        "candidate_id": item["sn_id"],
                        "target_stage": target_stage,
                        "attempt_count": 3,
                    }
                ]
            if "[stale:PRODUCED_NAME]" in cypher:
                return []
            return [{"id": row["sns_id"]} for row in params.get("batch", [])]

        self.atomic_tx.run.side_effect = _winner
        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=[candidate["id"]],
            ) as write,
            patch("imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"),
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                return_value=0,
            ),
        ):
            winners = persist_generated_name_batch(
                [candidate], compose_model="test/model", return_winner_ids=True
            )

        assert winners == ["dd:core_profiles/profiles_1d/electrons/temperature"]
        assert write.called is rich_write_expected
        cleanup_call = self.atomic_tx.run.call_args_list[1]
        cleanup_cypher = cleanup_call.args[0]
        cleanup_item = cleanup_call.kwargs["batch"][0]
        assert cleanup_item["preserve_current"] is True
        assert "old.id = b.sn_id" in cleanup_cypher
        assert "stale.claim_token = b.claim_token" in cleanup_cypher
        assert "stale.claim_seq = b.claim_seq" in cleanup_cypher
        assert "HAS_STANDARD_NAME" not in cleanup_cypher
        assert "source_paths =" not in cleanup_cypher
        finalize_cypher = _transaction_call(
            self.atomic_tx, "sns.status = 'composed'"
        ).args[0]
        assert finalize_cypher.count("sns.status = 'composed'") == 1
        assert "sns.last_error = null" in finalize_cypher
        assert "source_gap:HAS_STANDARD_NAME_VOCAB_GAP" in finalize_cypher
        assert "entity_gap:HAS_STANDARD_NAME_VOCAB_GAP" in finalize_cypher
        assert "FOREACH (edge IN source_gap_edges | DELETE edge)" in finalize_cypher
        assert "FOREACH (edge IN entity_gap_edges | DELETE edge)" in finalize_cypher
        assert "DELETE vg" not in finalize_cypher
        assert "HAS_EVIDENCE" not in finalize_cypher
        if binding_kind == "owned_reservation":
            assert "reservation.claim_token = b.claim_token" in finalize_cypher
            assert "reservation.claim_seq = b.claim_seq" in finalize_cypher
        else:
            assert "WHERE sn.name_stage IN $stable_stages" in finalize_cypher

    def test_terminal_transition_after_rich_write_rolls_back_every_effect(self):
        """A reservation that turns terminal before finalize cannot commit."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate()

        def _transition(cypher, **params):
            if "AS outcome" in cypher:
                item = params["batch"][0]
                return [
                    {
                        "id": item["sns_id"],
                        "outcome": "winner",
                        "binding_kind": "owned_reservation",
                        "candidate_id": item["sn_id"],
                        "target_stage": "pending",
                        "attempt_count": 1,
                    }
                ]
            if "reservation.provisional = true" in cypher:
                return []
            return [{"id": row["sns_id"]} for row in params.get("batch", [])]

        self.atomic_tx.run.side_effect = _transition
        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=[candidate["id"]],
            ) as write,
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"
            ) as backfill,
            pytest.raises(RuntimeError, match="source claim changed"),
        ):
            persist_generated_name_batch([candidate], compose_model="test/model")

        write.assert_called_once()
        backfill.assert_not_called()
        cleanup_cypher = self.atomic_tx.run.call_args_list[1].args[0]
        assert "b.preserve_current = true" in cleanup_cypher
        finalize_cypher = self.atomic_tx.run.call_args_list[2].args[0]
        assert "sn.name_stage = $pending_stage" in finalize_cypher
        assert "reservation.provisional = true" in finalize_cypher
        assert "reservation.claim_token = b.claim_token" in finalize_cypher
        assert "reservation.claim_seq = b.claim_seq" in finalize_cypher
        assert "MERGE (source)-[:HAS_STANDARD_NAME]->(sn)" in finalize_cypher
        self.atomic_tx.commit.assert_not_called()
        self.atomic_tx.close.assert_called_once()

    def test_reclaimed_collision_cleans_only_stamped_provisional_artifacts(self):
        """A reclaimed source drops stale staging residue, not real history."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate(name="hydrogen_fraction")

        def _collision(cypher, **params):
            if "AS outcome" not in cypher:
                return []
            item = params["batch"][0]
            return [
                {
                    "id": item["sns_id"],
                    "outcome": "lifecycle_collision",
                    "binding_kind": "collision",
                    "candidate_id": item["sn_id"],
                    "target_stage": "superseded",
                    "attempt_count": 4,
                }
            ]

        self.atomic_tx.run.side_effect = _collision
        with patch("imas_codex.standard_names.graph_ops.write_standard_names") as write:
            assert (
                persist_generated_name_batch(
                    [candidate],
                    compose_model="test/model",
                    return_winner_ids=True,
                )
                == []
            )

        write.assert_not_called()
        cleanup_cypher = self.atomic_tx.run.call_args_list[1].args[0]
        assert "stale.provisional = true" in cleanup_cypher
        assert "stale.claim_token IS NOT NULL" in cleanup_cypher
        assert "stale.claim_seq IS NOT NULL" in cleanup_cypher
        assert "collect(DISTINCT stale) AS stale_edges" in cleanup_cypher
        assert "edge.created_target = true" in cleanup_cypher
        assert "old.name_stage = $pending_stage" in cleanup_cypher
        assert "NOT EXISTS { MATCH (old)--() }" in cleanup_cypher
        assert "DELETE stale_edge" in cleanup_cypher
        assert "DELETE owned_target" in cleanup_cypher
        assert "HAS_STANDARD_NAME" not in cleanup_cypher
        assert "SET old.source_paths" not in cleanup_cypher
        release_cypher = self.atomic_tx.run.call_args_list[2].args[0]
        assert "sns.status = 'extracted'" in release_cypher
        assert "sns.claim_token = null" in release_cypher

    def test_unstamped_projection_and_source_paths_survive_stale_cleanup(self):
        """Provisional ownership never implies ownership of unstamped mirrors."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate(name="hydrogen_fraction")

        def _collision(cypher, **params):
            if "AS outcome" not in cypher:
                return []
            item = params["batch"][0]
            return [
                {
                    "id": item["sns_id"],
                    "outcome": "lifecycle_collision",
                    "binding_kind": "collision",
                    "candidate_id": item["sn_id"],
                    "target_stage": "superseded",
                    "attempt_count": 3,
                }
            ]

        self.atomic_tx.run.side_effect = _collision
        assert (
            persist_generated_name_batch(
                [candidate], compose_model="test/model", return_winner_ids=True
            )
            == []
        )

        cleanup_cypher = self.atomic_tx.run.call_args_list[1].args[0]
        assert "stale.provisional = true" in cleanup_cypher
        assert "DELETE stale_edge" in cleanup_cypher
        assert "HAS_STANDARD_NAME" not in cleanup_cypher
        assert "source_paths =" not in cleanup_cypher
        assert "coalesce(old.source_paths, []) = []" in cleanup_cypher

    def test_reclaimed_same_candidate_can_reserve_after_owned_orphan_cleanup(self):
        """An exact-owned pending orphan cannot wedge the next source claim."""
        from imas_codex.standard_names.graph_ops import _lock_claimed_name_bindings

        batch = [
            {
                "sns_id": "dd:core_profiles/electrons/temperature",
                "source_id": "core_profiles/electrons/temperature",
                "sn_id": "electron_temperature",
                "unit": "eV",
                "claim_token": "reclaimed",
                "claim_seq": 8,
            }
        ]
        gc = MagicMock()
        calls = {"lock": 0}

        def _query(cypher, **_params):
            if "AS outcome" in cypher:
                calls["lock"] += 1
                if calls["lock"] == 1:
                    return [
                        {
                            "id": batch[0]["sns_id"],
                            "outcome": "lifecycle_collision",
                            "binding_kind": "collision",
                            "candidate_id": batch[0]["sn_id"],
                            "target_stage": "pending",
                            "attempt_count": 2,
                        }
                    ]
                return [
                    {
                        "id": batch[0]["sns_id"],
                        "outcome": "winner",
                        "binding_kind": "owned_reservation",
                        "candidate_id": batch[0]["sn_id"],
                        "target_stage": "pending",
                        "attempt_count": 3,
                    }
                ]
            return []

        gc.query.side_effect = _query
        first = _lock_claimed_name_bindings(
            gc,
            batch,
            allow_missing=True,
            allow_own_pending_reservation=True,
        )
        second = _lock_claimed_name_bindings(
            gc,
            batch,
            allow_missing=True,
            allow_own_pending_reservation=True,
        )

        assert first[0]["outcome"] == "lifecycle_collision"
        assert second[0]["binding_kind"] == "owned_reservation"
        cleanup_cypher = gc.query.call_args_list[1].args[0]
        assert "edge.created_target = true" in cleanup_cypher
        assert "DELETE owned_target" in cleanup_cypher
        retry_lock = gc.query.call_args_list[3].args[0]
        assert "MERGE (target:StandardName {id: b.sn_id})" in retry_lock
        assert "reservation.created_target = true" in retry_lock

    def test_cleanup_reaps_multi_edge_owned_target_after_edge_deletion(self):
        """A target-owned stale group is reaped only after every edge is gone."""
        from imas_codex.standard_names.graph_ops import _lock_claimed_name_bindings

        current_source = "dd:core_profiles/electrons/temperature"
        retry_source = "dd:core_profiles/ions/temperature"
        nodes = {
            "hydrogen_fraction": {"stage": "pending", "source_paths": []},
            "electron_temperature": {"stage": "pending", "source_paths": []},
            "deuterium_fraction": {"stage": "pending", "source_paths": []},
        }
        edges = {
            "owned-first": {
                "source": current_source,
                "target": "hydrogen_fraction",
                "token": "abandoned-first",
                "seq": 5,
                "provisional": True,
                "created_target": True,
            },
            "owned-second": {
                "source": current_source,
                "target": "hydrogen_fraction",
                "token": "abandoned-second",
                "seq": 6,
                "provisional": True,
                "created_target": False,
            },
            "current-winner": {
                "source": current_source,
                "target": "electron_temperature",
                "token": "winner",
                "seq": 7,
                "provisional": True,
                "created_target": True,
            },
            "current-stale": {
                "source": current_source,
                "target": "electron_temperature",
                "token": "abandoned-current",
                "seq": 2,
                "provisional": True,
                "created_target": True,
            },
            "unowned-false": {
                "source": current_source,
                "target": "deuterium_fraction",
                "token": "unowned-first",
                "seq": 3,
                "provisional": True,
                "created_target": False,
            },
            "unowned-unset": {
                "source": current_source,
                "target": "deuterium_fraction",
                "token": "unowned-second",
                "seq": 4,
                "provisional": True,
                "created_target": None,
            },
        }
        deleted_edges = []
        deleted_targets = []

        def _query(cypher, **params):
            if "AS outcome" in cypher:
                results = []
                for item in params["batch"]:
                    target = nodes.get(item["sn_id"])
                    exact_edge = next(
                        (
                            edge
                            for edge in edges.values()
                            if edge["source"] == item["sns_id"]
                            and edge["target"] == item["sn_id"]
                            and edge["token"] == item["claim_token"]
                            and edge["seq"] == item["claim_seq"]
                        ),
                        None,
                    )
                    if target is None:
                        nodes[item["sn_id"]] = {
                            "stage": "pending",
                            "source_paths": [],
                        }
                        edge_id = f"reserved:{item['sns_id']}:{item['sn_id']}"
                        edges[edge_id] = {
                            "source": item["sns_id"],
                            "target": item["sn_id"],
                            "token": item["claim_token"],
                            "seq": item["claim_seq"],
                            "provisional": True,
                            "created_target": True,
                        }
                        exact_edge = edges[edge_id]
                    results.append(
                        {
                            "id": item["sns_id"],
                            "outcome": "winner",
                            "binding_kind": "owned_reservation",
                            "candidate_id": item["sn_id"],
                            "target_stage": nodes[item["sn_id"]]["stage"],
                            "attempt_count": 1,
                        }
                    )
                    assert exact_edge is not None
                return results

            if "[stale:PRODUCED_NAME]" not in cypher:
                return []

            assert "collect(DISTINCT stale) AS stale_edges" in cypher
            assert "any(edge IN stale_edges" in cypher
            assert "FOREACH (stale_edge IN stale_edges" in cypher
            assert "NOT EXISTS { MATCH (old)--() }" in cypher
            assert cypher.index("DELETE stale_edge") < cypher.index(
                "NOT EXISTS { MATCH (old)--() }"
            )
            for item in params["batch"]:
                stale_groups = {}
                for edge_id, edge in list(edges.items()):
                    if edge["source"] != item["sns_id"]:
                        continue
                    if not edge["provisional"]:
                        continue
                    if edge["token"] is None or edge["seq"] is None:
                        continue
                    is_current = (
                        item["preserve_current"]
                        and edge["target"] == item["sn_id"]
                        and edge["token"] == item["claim_token"]
                        and edge["seq"] == item["claim_seq"]
                    )
                    if not is_current:
                        stale_groups.setdefault(edge["target"], []).append(edge_id)

                for target_id, edge_ids in stale_groups.items():
                    owns_target = any(
                        edges[edge_id]["created_target"] is True for edge_id in edge_ids
                    )
                    for edge_id in edge_ids:
                        deleted_edges.append(edge_id)
                        del edges[edge_id]
                    target = nodes[target_id]
                    has_relationship = any(
                        edge["target"] == target_id for edge in edges.values()
                    )
                    if (
                        owns_target
                        and target["stage"] == params["pending_stage"]
                        and target["source_paths"] == []
                        and not has_relationship
                    ):
                        deleted_targets.append(target_id)
                        del nodes[target_id]
            return []

        gc = MagicMock()
        gc.query.side_effect = _query
        current_batch = [
            {
                "sns_id": current_source,
                "source_id": "core_profiles/electrons/temperature",
                "source_type": "dd",
                "sn_id": "electron_temperature",
                "unit": "eV",
                "claim_token": "winner",
                "claim_seq": 7,
            }
        ]
        current = _lock_claimed_name_bindings(
            gc,
            current_batch,
            allow_missing=True,
            allow_own_pending_reservation=True,
        )

        assert current[0]["binding_kind"] == "owned_reservation"
        assert {"owned-first", "owned-second"} <= set(deleted_edges)
        assert "hydrogen_fraction" in deleted_targets
        assert "hydrogen_fraction" not in nodes
        assert "current-stale" in deleted_edges
        assert "current-winner" in edges
        assert "electron_temperature" in nodes
        assert "electron_temperature" not in deleted_targets
        assert "deuterium_fraction" in nodes
        assert not any(
            edge["target"] == "deuterium_fraction" for edge in edges.values()
        )

        retry_batch = [
            {
                "sns_id": retry_source,
                "source_id": "core_profiles/ions/temperature",
                "source_type": "dd",
                "sn_id": "hydrogen_fraction",
                "unit": "1",
                "claim_token": "retry",
                "claim_seq": 9,
            }
        ]
        retry = _lock_claimed_name_bindings(
            gc,
            retry_batch,
            allow_missing=True,
            allow_own_pending_reservation=True,
        )

        assert retry[0]["binding_kind"] == "owned_reservation"
        assert "hydrogen_fraction" in nodes
        assert any(
            edge["source"] == retry_source
            and edge["target"] == "hydrogen_fraction"
            and edge["token"] == "retry"
            and edge["seq"] == 9
            for edge in edges.values()
        )

    def test_rich_write_and_finalize_share_transaction(self):
        """The rich writer and source outcome use the locked transaction."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        seen = {}

        def _write(names, *, gc, return_written_ids, **_kwargs):
            seen["write_gc"] = gc
            assert return_written_ids is True
            return [name["id"] for name in names]

        def _backfill(_names, *, gc, strict):
            seen["backfill_gc"] = gc
            assert strict is True

        def _supersede(_pairs, *, gc):
            seen["supersede_gc"] = gc
            return 0

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                side_effect=_write,
            ),
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources",
                side_effect=_backfill,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                side_effect=_supersede,
            ),
        ):
            result = persist_generated_name_batch(
                [_make_candidate()], compose_model="test/model"
            )

        assert result == 1
        assert seen["write_gc"] is seen["backfill_gc"] is seen["supersede_gc"]
        assert seen["write_gc"]._transaction is self.atomic_tx
        self.atomic_tx.commit.assert_called_once()

    def test_rich_write_failure_keeps_reservation_unfinalized(self):
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate()
        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                side_effect=RuntimeError("rich write failed"),
            ),
            patch(
                "imas_codex.standard_names.graph_ops._finalize_generated_name_stage"
            ) as finalize,
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"
            ) as backfill,
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names"
            ) as supersede,
            patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter") as counter,
        ):
            with pytest.raises(RuntimeError, match="rich write failed"):
                persist_generated_name_batch([candidate], compose_model="test/model")

        finalize.assert_not_called()
        backfill.assert_not_called()
        supersede.assert_not_called()
        counter.assert_not_called()
        self.atomic_tx.commit.assert_not_called()
        self.atomic_tx.close.assert_called_once()

    def test_post_finalize_failure_rolls_back_gap_retirement(self):
        """A later transactional failure retains gaps with every other write."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate()
        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=[candidate["id"]],
            ),
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources",
                side_effect=RuntimeError("cluster backfill failed"),
            ),
            pytest.raises(RuntimeError, match="cluster backfill failed"),
        ):
            persist_generated_name_batch([candidate], compose_model="test/model")

        finalize_cypher = _transaction_call(
            self.atomic_tx, "source_gap:HAS_STANDARD_NAME_VOCAB_GAP"
        ).args[0]
        assert "entity_gap:HAS_STANDARD_NAME_VOCAB_GAP" in finalize_cypher
        self.atomic_tx.commit.assert_not_called()
        self.atomic_tx.close.assert_called_once()

    def test_supersession_failure_rolls_back_generated_successor(self):
        """A fail-closed predecessor migration aborts the whole name write."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate()

        def _reject_supersession(_pairs, *, gc):
            assert gc._transaction is self.atomic_tx
            raise ValueError("generated-name attachment rejected")

        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=[candidate["id"]],
            ),
            patch("imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"),
            patch(
                "imas_codex.standard_names.graph_ops.supersede_prior_source_names",
                side_effect=_reject_supersession,
            ),
            patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter") as counter,
            pytest.raises(ValueError, match="attachment rejected"),
        ):
            persist_generated_name_batch([candidate], compose_model="test/model")

        self.atomic_tx.commit.assert_not_called()
        self.atomic_tx.close.assert_called_once()
        counter.assert_not_called()

    def test_partial_rich_write_keeps_reservation_unfinalized(self):
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = _make_candidate()
        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=0,
            ),
            patch(
                "imas_codex.standard_names.graph_ops._finalize_generated_name_stage"
            ) as finalize,
            patch(
                "imas_codex.standard_names.graph_ops._backfill_cluster_from_sources"
            ) as backfill,
        ):
            with pytest.raises(RuntimeError, match="rich standard-name write"):
                persist_generated_name_batch([candidate], compose_model="test/model")

        finalize.assert_not_called()
        backfill.assert_not_called()
        self.atomic_tx.commit.assert_not_called()
        self.atomic_tx.close.assert_called_once()

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
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops._finalize_generated_name_stage",
                return_value=["dd:core_profiles/profiles_1d/electrons/temperature"],
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

        mock_finalize.assert_not_called()
        finalize_batch = _transaction_call(
            self.atomic_tx, "sns.status = 'composed'"
        ).kwargs["batch"]
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


def _supersession_preflight(
    *,
    old_name: str | None = "old_pipeline_name",
    new_name: str = "new_pipeline_name",
    requested_source_id: str = "eq/q_95",
    trigger_source_id: str | None = "dd:eq/q_95",
    source_ids: list[str] | None = None,
    retained_source_ids: list[str] | None = None,
) -> dict[str, object]:
    """One preflight row.

    ``source_ids`` are the sources a composer already bound to the successor —
    the only ones eligible to carry their provenance across. ``retained_source_ids``
    are the sources that reach the predecessor and nothing else; they hold it
    live.
    """
    return {
        "requested_source_id": requested_source_id,
        "new_name": new_name,
        "requested_source_exists": True,
        "successor_exists": True,
        "trigger_source_id": trigger_source_id,
        "old_name": old_name,
        "old_stage": "accepted" if old_name else None,
        "judged_source_ids": (source_ids if source_ids is not None else ["dd:eq/q_95"]),
        "retained_source_ids": retained_source_ids or [],
    }


def _supersession_graph(
    preflight_rows: list[dict[str, object]],
    *,
    finalized_rows: list[dict[str, str]] | None = None,
):
    tx = MagicMock()
    tx.closed = False

    def _run(cypher, **params):
        if "GENERATED_SUPERSESSION_PREFLIGHT" in cypher:
            return preflight_rows
        if "GENERATED_SUPERSESSION_FINALIZE" in cypher:
            if finalized_rows is not None:
                return finalized_rows
            return [
                {
                    "old_name": plan["old_name"],
                    "new_name": plan["new_name"],
                }
                for plan in params["plans"]
            ]
        raise AssertionError(f"unexpected supersession query: {cypher}")

    tx.run.side_effect = _run
    session = MagicMock()
    session.begin_transaction.return_value = tx

    @contextmanager
    def _session():
        yield session

    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = False
    graph.session = _session
    return graph, tx


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
        from imas_codex.standard_names.attachment_audit import (
            AttachmentPairingGuardResult,
        )
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        graph, tx = _supersession_graph([_supersession_preflight()])
        admitted = AttachmentPairingGuardResult(("dd:eq/q_95",), ())
        with (
            _patch_gc(graph),
            patch(
                "imas_codex.standard_names.attachment_audit.guard_source_pairings",
                return_value=admitted,
            ),
            patch(
                "imas_codex.standard_names.provenance_lifecycle."
                "retarget_standard_name_sources",
                return_value=1,
            ),
        ):
            n = supersede_prior_source_names(
                [{"new_name": "new_pipeline_name", "source_id": "eq/q_95"}]
            )

        assert n == 1
        cypher = _transaction_call(tx, "old.name_stage = 'superseded'").args[0]
        # The predecessor is marked superseded and linked via REFINED_FROM.
        assert "old.name_stage = 'superseded'" in cypher
        assert "MERGE (new)-[:REFINED_FROM]->(old)" in cypher
        # A structural parent is exempt by origin: it belongs to the admission
        # gate, not to any one source. An imported name is NOT exempt by origin
        # alone — the graph plus the review pipeline is the source of truth, so
        # an unreviewed import must not keep competing for a source that has
        # already recomposed. Publication through a merged catalog PR is what
        # exempts it, pinned in test_supersession_catalog_guard.
        assert "coalesce(old.origin, 'pipeline') <> 'derived'" in cypher
        assert "catalog_edit" not in cypher
        # Already-retired / frozen / published names are never re-superseded.
        assert "['superseded', 'exhausted', 'contested', 'approved']" in cypher
        # The new name itself is never superseded (byte-identical regen no-op).
        preflight_cypher = _transaction_call(
            tx, "GENERATED_SUPERSESSION_PREFLIGHT"
        ).args[0]
        assert "old.id <> pr.new_name" in preflight_cypher

    def test_open_edit_propagated_to_successor_and_predecessor_reconciled(self):
        """A name-hint regen that supersedes the edited predecessor must ride
        the still-open edit forward onto the recomposed successor and reconcile
        the predecessor to 'applied' — otherwise the edit is stuck 'open' on a
        superseded node forever."""
        from imas_codex.standard_names.attachment_audit import (
            AttachmentPairingGuardResult,
        )
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        graph, tx = _supersession_graph(
            [_supersession_preflight(old_name="old_name", new_name="new_name")]
        )
        admitted = AttachmentPairingGuardResult(("dd:eq/q_95",), ())
        with (
            _patch_gc(graph),
            patch(
                "imas_codex.standard_names.attachment_audit.guard_source_pairings",
                return_value=admitted,
            ),
            patch(
                "imas_codex.standard_names.provenance_lifecycle."
                "retarget_standard_name_sources",
                return_value=1,
            ),
        ):
            supersede_prior_source_names(
                [{"new_name": "new_name", "source_id": "eq/q_95"}]
            )

        cypher = _transaction_call(tx, "carry_edit").args[0]
        # The propagation is gated on the predecessor's edit still being open.
        assert "(coalesce(old.edit_status, '') = 'open') AS carry_edit" in cypher
        # Predecessor reconciled to 'applied', not left stuck 'open'.
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

        graph, tx = _supersession_graph(
            [_supersession_preflight(old_name=None, new_name="same_name")]
        )
        with _patch_gc(graph):
            n = supersede_prior_source_names(
                [{"new_name": "same_name", "source_id": "eq/q_95"}]
            )

        assert n == 0
        tx.commit.assert_called_once_with()

    def test_guard_rejection_rolls_back_before_any_mutation(self):
        from imas_codex.standard_names.attachment_audit import (
            AttachmentPairingGuardResult,
            AttachmentVerdict,
        )
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        graph, tx = _supersession_graph([_supersession_preflight()])
        rejected = AttachmentVerdict(
            "dd:eq/q_95",
            "eq/q_95",
            "new_pipeline_name",
            "drafted",
            "unit dimensionality mismatch",
        )
        guard_result = AttachmentPairingGuardResult((), (rejected,))
        with (
            _patch_gc(graph),
            patch(
                "imas_codex.standard_names.attachment_audit.guard_source_pairings",
                return_value=guard_result,
            ),
            patch(
                "imas_codex.standard_names.provenance_lifecycle."
                "retarget_standard_name_sources"
            ) as retarget,
            pytest.raises(ValueError, match="supersession rolled back"),
        ):
            supersede_prior_source_names(
                [{"new_name": "new_pipeline_name", "source_id": "eq/q_95"}]
            )

        tx.rollback.assert_called_once_with()
        tx.commit.assert_not_called()
        retarget.assert_not_called()
        assert not any(
            "GENERATED_SUPERSESSION_FINALIZE" in call.args[0]
            for call in tx.run.call_args_list
        )

    def test_partial_source_move_rolls_back_without_lifecycle_effects(self):
        from imas_codex.standard_names.attachment_audit import (
            AttachmentPairingGuardResult,
        )
        from imas_codex.standard_names.graph_ops import supersede_prior_source_names

        source_ids = ["dd:eq/q_95", "dd:eq/q_axis"]
        graph, tx = _supersession_graph(
            [_supersession_preflight(source_ids=source_ids)]
        )
        admitted = AttachmentPairingGuardResult(tuple(source_ids), ())
        with (
            _patch_gc(graph),
            patch(
                "imas_codex.standard_names.attachment_audit.guard_source_pairings",
                return_value=admitted,
            ),
            patch(
                "imas_codex.standard_names.provenance_lifecycle."
                "retarget_standard_name_sources",
                return_value=1,
            ) as retarget,
            pytest.raises(RuntimeError, match="expected 2, moved 1"),
        ):
            supersede_prior_source_names(
                [{"new_name": "new_pipeline_name", "source_id": "eq/q_95"}]
            )

        tx.rollback.assert_called_once_with()
        tx.commit.assert_not_called()
        assert retarget.call_args.kwargs["source_ids"] == source_ids
        assert not any(
            "GENERATED_SUPERSESSION_FINALIZE" in call.args[0]
            for call in tx.run.call_args_list
        )

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

        def _fake_supersede(pairs, **_kwargs):
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

        gc, _tx = _mock_gc_tx()
        with (
            patch(
                "imas_codex.standard_names.graph_ops.write_standard_names",
                return_value=1,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=gc,
            ),
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

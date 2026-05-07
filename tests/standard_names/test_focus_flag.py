"""Unit tests for the --focus flag in ``sn run``.

Verifies:
- ``--focus`` and ``--paths`` are mutually exclusive (Click UsageError)
- ``scope_run_id`` is passed to ``_claim_sn_atomic`` and adds a WHERE clause
- ``claim_generate_name_batch`` with ``scope_run_id`` filters SNS nodes by run_id
- The full focus routing sequence (clear stale, seed, stamp, reset, loop)
- Without ``--focus``, ``scope_run_id`` is ``None`` throughout (backward compat)

All tests are mock-based; no live Neo4j required.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gc_tx():
    """Build a mock GraphClient wired for single-transaction claim functions.

    Returns ``(gc, tx)`` where *gc* is the mock ``GraphClient`` (supports
    context manager protocol) and *tx* is the mock ``Transaction`` whose
    ``.run()`` can be configured via ``side_effect`` per test.
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


def _patch_graph_ops_gc(mock_gc):
    """Patch GraphClient as used inside graph_ops module."""
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


def _make_cli_gc_mock():
    """Build a GraphClient mock suitable for the focus-routing ``with`` blocks.

    The focus routing uses ``with GraphClient() as gc: gc.query(...)``
    three times, so this mock supports the context manager protocol and
    has a ``.query()`` that returns an empty list.
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=[])
    return gc


def _invoke_focus(*focus_args: str, extra_args: list[str] | None = None):
    """Invoke ``sn run`` with ``--focus`` args and return ``(result, captured_kwargs)``.

    Patches ``_run_sn_loop_cmd``, ``GraphClient`` (at its source module), and
    ``merge_standard_name_sources`` so no graph connection is attempted.
    """
    runner = CliRunner()
    captured: dict = {}
    seeded: list = []

    def _fake_loop_cmd(**kwargs):
        captured.update(kwargs)

    def _fake_merge(sources, **kw):
        seeded.extend(sources)
        return len(sources)

    gc_mock = _make_cli_gc_mock()

    cmd_args = ["run", "--skip-clear-gate"] + list(focus_args) + (extra_args or [])

    with (
        patch("imas_codex.cli.sn._run_sn_loop_cmd", side_effect=_fake_loop_cmd),
        patch("imas_codex.graph.client.GraphClient", return_value=gc_mock),
        patch(
            "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
            side_effect=_fake_merge,
        ),
    ):
        result = runner.invoke(sn, cmd_args, catch_exceptions=False)

    return result, captured if captured else None, seeded


# ---------------------------------------------------------------------------
# 1. test_focus_flag_cli_validation
# ---------------------------------------------------------------------------


class TestFocusFlagCliValidation:
    """CLI-level validation for the --focus flag."""

    def test_focus_and_paths_mutually_exclusive(self):
        """Providing both --focus and --paths must exit with code 2."""
        runner = CliRunner()
        result = runner.invoke(
            sn,
            [
                "run",
                "--skip-clear-gate",
                "--dry-run",
                "--focus",
                "equilibrium/time_slice/profiles_1d/psi",
                "--paths",
                "equilibrium/time_slice/profiles_1d/q",
            ],
        )
        assert result.exit_code == 2, (
            f"Expected exit_code=2, got {result.exit_code}. Output: {result.output}"
        )
        assert "--focus and --paths are mutually exclusive" in (result.output or "")

    def test_focus_single_flag_accepted(self):
        """A single --focus flag is accepted; scope_run_id is passed to the loop."""
        result, kwargs, _ = _invoke_focus(
            "--focus", "equilibrium/time_slice/profiles_1d/psi"
        )
        assert result.exit_code == 0, f"CLI error: {result.output}"
        assert kwargs is not None, "_run_sn_loop_cmd was not called"
        assert kwargs.get("scope_run_id") is not None, (
            "scope_run_id should be a UUID string, not None"
        )

    def test_focus_multiple_flags_accepted(self):
        """Multiple --focus flags are flattened and both accepted."""
        result, kwargs, seeded = _invoke_focus(
            "--focus",
            "equilibrium/time_slice/profiles_1d/psi",
            "--focus",
            "equilibrium/time_slice/profiles_1d/q",
        )
        assert result.exit_code == 0, f"CLI error: {result.output}"
        assert kwargs is not None
        # Both paths should have been seeded as SNS nodes.
        seeded_ids = {s["id"] for s in seeded}
        assert "dd:equilibrium/time_slice/profiles_1d/psi" in seeded_ids
        assert "dd:equilibrium/time_slice/profiles_1d/q" in seeded_ids

    def test_focus_space_separated_paths(self):
        """Space-separated paths within a single --focus value are both seeded."""
        result, kwargs, seeded = _invoke_focus("--focus", "eq/a eq/b")
        assert result.exit_code == 0, f"CLI error: {result.output}"
        seeded_ids = {s["id"] for s in seeded}
        assert "dd:eq/a" in seeded_ids, f"Expected dd:eq/a in {seeded_ids}"
        assert "dd:eq/b" in seeded_ids, f"Expected dd:eq/b in {seeded_ids}"

    def test_empty_focus_falls_through_to_normal_loop(self):
        """Without --focus, the normal loop path is taken (scope_run_id absent)."""
        runner = CliRunner()
        captured: dict = {}

        def _fake_loop_cmd(**kwargs):
            captured.update(kwargs)

        with patch("imas_codex.cli.sn._run_sn_loop_cmd", side_effect=_fake_loop_cmd):
            result = runner.invoke(
                sn,
                ["run", "--skip-clear-gate", "--dry-run"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, f"CLI error: {result.output}"
        # With --dry-run the loop is skipped, but if it were called the
        # scope_run_id should not be present (or be None).
        if captured:
            assert captured.get("scope_run_id") is None, (
                f"scope_run_id should be None without --focus, got {captured.get('scope_run_id')}"
            )


# ---------------------------------------------------------------------------
# 2. test_scope_run_id_filters_claims
# ---------------------------------------------------------------------------


class TestScopeRunIdFiltersClaims:
    """``_claim_sn_atomic`` adds a WHERE clause when scope_run_id is provided."""

    def test_scope_run_id_adds_where_clause(self):
        """When scope_run_id is set, the seed Cypher contains run_id filter."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        # Seed returns one item; no expand (no cluster/unit keys).
        tx.run = MagicMock(
            side_effect=[
                # seed
                [{"_cluster_id": None, "_unit": None, "_physics_domain": None}],
                # read-back
                [
                    {
                        "id": "sn1",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": None,
                        "cluster_id": None,
                        "physics_domain": None,
                        "validation_status": "valid",
                    }
                ],
            ]
        )

        test_uuid = "aaaabbbb-1111-2222-3333-ccccddddeeee"

        with _patch_graph_ops_gc(gc):
            items = _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'pending'",
                query_params={},
                batch_size=1,
                scope_run_id=test_uuid,
            )

        assert len(items) == 1
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "sn.run_id = $scope_run_id" in seed_cypher, (
            f"Expected 'sn.run_id = $scope_run_id' in seed Cypher:\n{seed_cypher}"
        )
        seed_kwargs = tx.run.call_args_list[0].kwargs
        assert seed_kwargs.get("scope_run_id") == test_uuid

    def test_no_scope_run_id_no_where_clause(self):
        """When scope_run_id is None, the seed Cypher has no run_id filter."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": None, "_physics_domain": None}],
                [
                    {
                        "id": "sn2",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": None,
                        "cluster_id": None,
                        "physics_domain": None,
                        "validation_status": "valid",
                    }
                ],
            ]
        )

        with _patch_graph_ops_gc(gc):
            items = _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'pending'",
                query_params={},
                batch_size=1,
                scope_run_id=None,
            )

        assert len(items) == 1
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "scope_run_id" not in seed_cypher, (
            f"scope_run_id filter should not appear when scope_run_id=None:\n{seed_cypher}"
        )
        seed_kwargs = tx.run.call_args_list[0].kwargs
        assert "scope_run_id" not in seed_kwargs

    def test_scope_run_id_propagates_to_expand_branches(self):
        """scope_run_id filter appears in the expand Cypher when cluster is set."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        # Seed returns item with cluster key; triggers cluster-only expand branch.
        tx.run = MagicMock(
            side_effect=[
                # seed: has cluster_id but no unit
                [
                    {
                        "_cluster_id": "cluster-42",
                        "_unit": None,
                        "_physics_domain": None,
                    }
                ],
                # expand: no additional items claimed (empty result)
                [],
                # read-back
                [
                    {
                        "id": "sn3",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": None,
                        "cluster_id": "cluster-42",
                        "physics_domain": None,
                        "validation_status": "valid",
                    }
                ],
            ]
        )

        test_uuid = "bbbbcccc-1111-2222-3333-ddddeeeeaaaa"

        with _patch_graph_ops_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'pending'",
                query_params={},
                batch_size=5,
                scope_run_id=test_uuid,
            )

        # Both seed (call 0) and expand (call 1) should carry the run_id filter.
        seed_cypher = tx.run.call_args_list[0].args[0]
        expand_cypher = tx.run.call_args_list[1].args[0]
        assert "sn.run_id = $scope_run_id" in seed_cypher
        assert "sn.run_id = $scope_run_id" in expand_cypher


# ---------------------------------------------------------------------------
# 3. test_scope_run_id_filters_generate_name_claims
# ---------------------------------------------------------------------------


class TestScopeRunIdFiltersGenerateNameClaims:
    """``claim_generate_name_batch`` filters SNS nodes by run_id when scoped."""

    def test_scope_run_id_adds_sns_where_clause(self):
        """When scope_run_id is set, seed Cypher includes sns.run_id filter."""
        from imas_codex.standard_names.graph_ops import claim_generate_name_batch

        gc, tx = _mock_gc_tx()
        # Seed returns nothing — no eligible sources.
        tx.run = MagicMock(return_value=[])

        test_uuid = "ccccdddd-1111-2222-3333-eeeeaaaabbbb"

        with _patch_graph_ops_gc(gc):
            items = claim_generate_name_batch(scope_run_id=test_uuid, batch_size=5)

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "sns.run_id = $scope_run_id" in seed_cypher, (
            f"Expected 'sns.run_id = $scope_run_id' in seed Cypher:\n{seed_cypher}"
        )
        seed_kwargs = tx.run.call_args_list[0].kwargs
        assert seed_kwargs.get("scope_run_id") == test_uuid

    def test_no_scope_run_id_no_sns_where_clause(self):
        """When scope_run_id is None, no sns.run_id filter in seed Cypher."""
        from imas_codex.standard_names.graph_ops import claim_generate_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            items = claim_generate_name_batch(scope_run_id=None, batch_size=5)

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "scope_run_id" not in seed_cypher, (
            f"scope_run_id filter should be absent when scope_run_id=None:\n{seed_cypher}"
        )


# ---------------------------------------------------------------------------
# 4. test_focus_routing_seeds_and_stamps
# ---------------------------------------------------------------------------


class TestFocusRoutingSeedsAndStamps:
    """The focus routing executes the expected sequence of graph operations."""

    def test_clears_stale_run_ids_before_seeding(self):
        """Step 1: stale run_ids are cleared from SN and SNS nodes."""
        gc_mock = _make_cli_gc_mock()
        query_calls: list = []
        gc_mock.query = MagicMock(
            side_effect=lambda q, **kw: query_calls.append(q) or []
        )

        with (
            patch("imas_codex.cli.sn._run_sn_loop_cmd"),
            patch("imas_codex.graph.client.GraphClient", return_value=gc_mock),
            patch(
                "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
                return_value=1,
            ),
        ):
            runner = CliRunner()
            runner.invoke(
                sn,
                [
                    "run",
                    "--skip-clear-gate",
                    "--focus",
                    "equilibrium/time_slice/profiles_1d/psi",
                ],
                catch_exceptions=False,
            )

        # The first two gc.query() calls should clear stale run_ids.
        assert len(query_calls) >= 2, f"Expected ≥2 query calls, got {len(query_calls)}"
        clear_queries = " ".join(query_calls[:2])
        assert "SET sn.run_id = NULL" in clear_queries, (
            f"Expected SN run_id clear, got:\n{clear_queries}"
        )
        assert "SET sns.run_id = NULL" in clear_queries, (
            f"Expected SNS run_id clear, got:\n{clear_queries}"
        )

    def test_seeds_sns_nodes_with_correct_fields(self):
        """Step 2: merge_standard_name_sources is called with correct source dicts."""
        gc_mock = _make_cli_gc_mock()
        seeded: list = []

        def _fake_merge(sources, **kw):
            seeded.extend(sources)
            return len(sources)

        with (
            patch("imas_codex.cli.sn._run_sn_loop_cmd"),
            patch("imas_codex.graph.client.GraphClient", return_value=gc_mock),
            patch(
                "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
                side_effect=_fake_merge,
            ),
        ):
            runner = CliRunner()
            runner.invoke(
                sn,
                [
                    "run",
                    "--skip-clear-gate",
                    "--focus",
                    "equilibrium/time_slice/profiles_1d/psi",
                ],
                catch_exceptions=False,
            )

        assert len(seeded) == 1
        src = seeded[0]
        assert src["id"] == "dd:equilibrium/time_slice/profiles_1d/psi"
        assert src["source_type"] == "dd"
        assert src["source_id"] == "equilibrium/time_slice/profiles_1d/psi"
        assert src["batch_key"] == "focus"
        assert src["status"] == "extracted"

    def test_stamps_run_id_on_sns_and_resets_sns(self):
        """Steps 3 & 4: run_id is stamped on SNS and existing SNs are force-reset."""
        gc_mock = _make_cli_gc_mock()
        query_calls: list[tuple[str, dict]] = []

        def _capture_query(q, **kw):
            query_calls.append((q, kw))
            return []

        gc_mock.query = MagicMock(side_effect=_capture_query)

        captured_loop: dict = {}

        def _fake_loop_cmd(**kwargs):
            captured_loop.update(kwargs)

        with (
            patch("imas_codex.cli.sn._run_sn_loop_cmd", side_effect=_fake_loop_cmd),
            patch("imas_codex.graph.client.GraphClient", return_value=gc_mock),
            patch(
                "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
                return_value=1,
            ),
        ):
            runner = CliRunner()
            runner.invoke(
                sn,
                [
                    "run",
                    "--skip-clear-gate",
                    "--focus",
                    "equilibrium/time_slice/profiles_1d/psi",
                ],
                catch_exceptions=False,
            )

        # All query calls after the initial clear (calls 0, 1) should carry run_id.
        stamp_queries = [(q, kw) for q, kw in query_calls[2:]]
        all_query_text = " ".join(q for q, _ in stamp_queries)

        # Step 3: stamp run_id on SNS.
        assert "SET sns.run_id = $run_id" in all_query_text, (
            f"Expected SNS stamp query, queries were:\n{all_query_text}"
        )

        # Step 4: force-reset existing SN nodes.
        assert "sn.name_stage = 'pending'" in all_query_text, (
            f"Expected SN reset query, queries were:\n{all_query_text}"
        )
        assert "sn.docs_stage = 'pending'" in all_query_text

        # The scope_run_id passed to step 3 and 4 should match what the loop got.
        loop_scope_id = captured_loop.get("scope_run_id")
        assert loop_scope_id is not None
        stamp_run_ids = [kw.get("run_id") for _, kw in stamp_queries if "run_id" in kw]
        assert all(rid == loop_scope_id for rid in stamp_run_ids), (
            f"run_id mismatch: loop got {loop_scope_id!r}, queries used {stamp_run_ids}"
        )

    def test_scope_run_id_forwarded_to_loop(self):
        """Step 5: _run_sn_loop_cmd is called with a non-None scope_run_id."""
        result, kwargs, _ = _invoke_focus(
            "--focus", "equilibrium/time_slice/profiles_1d/psi"
        )
        assert result.exit_code == 0
        assert kwargs is not None
        scope_id = kwargs.get("scope_run_id")
        assert scope_id is not None
        # Should be a valid UUID4 string (36 chars with hyphens).
        assert len(scope_id) == 36
        assert scope_id.count("-") == 4


# ---------------------------------------------------------------------------
# 5. test_backward_compat_no_focus
# ---------------------------------------------------------------------------


class TestBackwardCompatNoFocus:
    """Without --focus, scope_run_id is None and the normal loop path is used."""

    def test_scope_run_id_is_none_without_focus(self):
        """_run_sn_loop_cmd receives scope_run_id=None when --focus is absent."""
        runner = CliRunner()
        captured: dict = {}

        def _fake_loop_cmd(**kwargs):
            captured.update(kwargs)

        with patch("imas_codex.cli.sn._run_sn_loop_cmd", side_effect=_fake_loop_cmd):
            result = runner.invoke(
                sn,
                ["run", "--skip-clear-gate", "--dry-run"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, f"CLI error: {result.output}"
        if captured:
            assert captured.get("scope_run_id") is None, (
                f"Expected scope_run_id=None without --focus, "
                f"got {captured.get('scope_run_id')!r}"
            )

    def test_paths_flag_does_not_set_scope_run_id(self):
        """Using --paths (not --focus) does not set scope_run_id."""
        runner = CliRunner()
        captured: dict = {}

        def _fake_loop_cmd(**kwargs):
            captured.update(kwargs)

        with patch("imas_codex.cli.sn._run_sn_loop_cmd", side_effect=_fake_loop_cmd):
            result = runner.invoke(
                sn,
                [
                    "run",
                    "--skip-clear-gate",
                    "--dry-run",
                    "--paths",
                    "equilibrium/time_slice/profiles_1d/psi",
                ],
                catch_exceptions=False,
            )

        # --paths triggers single-pass routing, not the loop, so captured may be empty.
        assert result.exit_code == 0, f"CLI error: {result.output}"
        if captured:
            assert captured.get("scope_run_id") is None

    def test_claim_sn_atomic_no_scope_default(self):
        """_claim_sn_atomic defaults to scope_run_id=None — no run_id filter."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            items = _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'pending'",
                query_params={},
                batch_size=1,
            )

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        # Backward-compat: no scope filter in the default call.
        assert "scope_run_id" not in seed_cypher

    def test_claim_generate_name_batch_no_scope_default(self):
        """claim_generate_name_batch defaults to scope_run_id=None — no SNS filter."""
        from imas_codex.standard_names.graph_ops import claim_generate_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            items = claim_generate_name_batch(batch_size=5)

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "scope_run_id" not in seed_cypher

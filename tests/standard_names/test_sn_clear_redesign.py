"""Tests for `sn clear` and `sn prune`.

`sn clear` is an unconditional full subsystem wipe followed by an auto
grammar re-seed (skippable with `--no-reseed`); scoped deletes live in
`sn prune`. The standalone grammar-sync command has been retired —
grammar is auto-synced at `sn run` startup and re-seeded by `sn clear`.
The underlying graph op (`clear_sn_subsystem`) never touches grammar
labels itself; the re-seed is driven by the `sn clear` CLI wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


class TestSnClearHelp:
    """`sn clear --help` should reflect the full-wipe redesign."""

    def _help(self) -> str:
        runner = CliRunner()
        result = runner.invoke(sn, ["clear", "--help"])
        assert result.exit_code == 0
        return result.output

    def test_clear_help_mentions_full_wipe(self):
        txt = self._help()
        assert "subsystem" in txt.lower() or "wipe" in txt.lower()

    def test_clear_help_has_dry_run(self):
        assert "--dry-run" in self._help()

    def test_clear_help_has_no_reseed_flag(self):
        # Clear auto-re-seeds the grammar; --no-reseed opts out.
        assert "--no-reseed" in self._help()

    def test_clear_help_does_not_mention_sync_grammar(self):
        # The standalone sync-grammar command is retired — clear handles
        # the re-seed itself.
        assert "sync-grammar" not in self._help()

    def test_clear_help_has_no_status_flag(self):
        # Scoped flags must have moved to `sn prune`.
        assert "--status" not in self._help()

    def test_clear_help_has_no_source_flag(self):
        assert "--source" not in self._help()

    def test_clear_help_has_no_ids_flag(self):
        assert "--ids" not in self._help()


class TestSnPruneHelp:
    """`sn prune` is the new scoped-delete tool."""

    def _help(self) -> str:
        runner = CliRunner()
        result = runner.invoke(sn, ["prune", "--help"])
        assert result.exit_code == 0
        return result.output

    def test_prune_registered(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert "prune" in result.output

    def test_prune_has_status(self):
        assert "--status" in self._help()

    def test_prune_has_all(self):
        assert "--all" in self._help()

    def test_prune_has_source(self):
        assert "--source" in self._help()

    def test_prune_has_ids(self):
        assert "--ids" in self._help()

    def test_prune_has_include_accepted(self):
        assert "--include-accepted" in self._help()

    def test_prune_has_include_sources(self):
        assert "--include-sources" in self._help()


class TestSnSyncGrammarRetired:
    """The standalone `sn sync-grammar` command has been retired."""

    def test_sync_grammar_not_registered(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert "sync-grammar" not in result.output

    def test_sync_grammar_invocation_unknown(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["sync-grammar", "--help"])
        assert result.exit_code != 0


class TestClearSnSubsystemLabels:
    """`clear_sn_subsystem` must touch every SN-pipeline-output label.

    Grammar labels (GrammarToken, GrammarSegment, GrammarTemplate,
    ISNGrammarVersion) are ISN-authoritative reference data and are
    NEVER touched by the `clear_sn_subsystem` graph op — the `sn clear`
    CLI wrapper re-seeds them after the wipe.
    """

    _EXPECTED_LABELS = {
        "StandardName",
        "StandardNameReview",
        "StandardNameSource",
        "DocsRevision",
        "VocabGap",
        "SNRun",
        "LLMCost",
    }

    _GRAMMAR_LABELS = {
        "GrammarToken",
        "GrammarSegment",
        "GrammarTemplate",
        "ISNGrammarVersion",
    }

    def test_counts_only_pipeline_output_labels(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            result = graph_ops.clear_sn_subsystem(dry_run=True)

        assert set(result.keys()) == self._EXPECTED_LABELS
        # Must not touch grammar labels.
        assert not (set(result.keys()) & self._GRAMMAR_LABELS)

    def test_dry_run_does_not_touch_graph(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=True)

        queries = [call.args[0] for call in fake_gc.query.call_args_list]
        assert not any("DETACH DELETE" in q for q in queries)

    def test_wipe_deletes_only_pipeline_labels(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 5}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=False)

        queries = [call.args[0] for call in fake_gc.query.call_args_list]
        detach_deletes = [q for q in queries if "DETACH DELETE" in q]
        # One DETACH DELETE per pipeline label — NOT 9.
        assert len(detach_deletes) == len(self._EXPECTED_LABELS)
        for label in self._EXPECTED_LABELS:
            assert any(label in q for q in detach_deletes), (
                f"Missing DETACH DELETE for {label}"
            )
        # Must NOT issue DETACH DELETE on any grammar label.
        for label in self._GRAMMAR_LABELS:
            assert not any(label in q for q in detach_deletes), (
                f"clear_sn_subsystem should not touch grammar label {label}"
            )

    def test_graph_op_never_calls_grammar_sync(self):
        """The `clear_sn_subsystem` graph op itself never re-seeds grammar.

        The re-seed is the responsibility of the `sn clear` CLI wrapper,
        not the low-level graph op (so callers that only want the wipe
        get exactly that).
        """
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with (
            patch.object(graph_ops, "GraphClient", return_value=fake_gc),
            patch(
                "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
            ) as mock_sync,
        ):
            graph_ops.clear_sn_subsystem(dry_run=False)

        mock_sync.assert_not_called()


class TestClearCliReseed:
    """The `sn clear` CLI wrapper re-seeds grammar after the wipe."""

    def _patch_clear(self, preview_count: int):
        """Patch clear_sn_subsystem so it reports/deletes ``preview_count`` nodes."""
        from imas_codex.standard_names import graph_ops

        def _fake_clear(*, dry_run: bool):
            return {"StandardName": preview_count}

        return patch.object(graph_ops, "clear_sn_subsystem", side_effect=_fake_clear)

    def test_clear_reseeds_grammar_by_default(self):
        runner = CliRunner()
        with (
            self._patch_clear(3),
            patch(
                "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
            ) as mock_sync,
        ):
            mock_sync.return_value = MagicMock(
                isn_version="9.9.9", segments=11, templates=6
            )
            result = runner.invoke(sn, ["clear", "--force", "--no-comment-export"])
        assert result.exit_code == 0, result.output
        mock_sync.assert_called_once()

    def test_no_reseed_skips_grammar_sync(self):
        runner = CliRunner()
        with (
            self._patch_clear(3),
            patch(
                "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
            ) as mock_sync,
        ):
            result = runner.invoke(
                sn,
                ["clear", "--force", "--no-comment-export", "--no-reseed"],
            )
        assert result.exit_code == 0, result.output
        mock_sync.assert_not_called()

    def test_review_deleted_before_standardname(self):
        """Pre-p39 bug: deleting StandardName first left orphan StandardNameReview nodes.

        The order in `clear_sn_subsystem` must be StandardNameReview → StandardName so
        even in the absence of HAS_STANDARD_NAME edges (pathological data)
        no orphan StandardNameReviews are left behind.
        """
        from imas_codex.standard_names import graph_ops

        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None
        fake_gc.query = MagicMock(return_value=[{"n": 0}])

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=False)

        queries = [call.args[0] for call in fake_gc.query.call_args_list]
        review_idx = next(
            i
            for i, q in enumerate(queries)
            if "StandardNameReview" in q and "DELETE" in q
        )
        sn_idx = next(
            i
            for i, q in enumerate(queries)
            if "StandardName" in q
            and "DELETE" in q
            and "Source" not in q
            and "Review" not in q
        )
        assert review_idx < sn_idx

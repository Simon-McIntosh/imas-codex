"""Tests for the shared reset infrastructure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _import_reset():
    """Import reset module without triggering discovery.__init__ (which needs litellm)."""
    import importlib.util
    import pathlib

    # Navigate: tests/discovery/test_reset.py -> ../../imas_codex/discovery/base/reset.py
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    mod_path = repo_root / "imas_codex" / "discovery" / "base" / "reset.py"

    import sys

    spec_obj = importlib.util.spec_from_file_location("_reset", str(mod_path))
    mod = importlib.util.module_from_spec(spec_obj)
    sys.modules["_reset"] = mod  # dataclass decorator needs this
    spec_obj.loader.exec_module(mod)
    return mod


@pytest.fixture
def reset_mod():
    return _import_reset()


# ─── ResetSpec tests ─────────────────────────────────────────────────────


class TestResetSpec:
    """Test ResetSpec dataclass and registry."""

    def test_all_domains_registered(self, reset_mod):
        assert set(reset_mod.DOMAIN_RESET_SPECS.keys()) == {
            "signals",
            "paths",
            "wiki",
            "code",
            "documents",
        }

    def test_get_valid_targets_signals(self, reset_mod):
        assert reset_mod.get_valid_targets("signals") == ["discovered", "enriched"]

    def test_get_valid_targets_paths(self, reset_mod):
        assert reset_mod.get_valid_targets("paths") == ["scanned", "triaged"]

    def test_get_valid_targets_wiki(self, reset_mod):
        assert reset_mod.get_valid_targets("wiki") == ["scanned", "scored"]

    def test_get_valid_targets_code(self, reset_mod):
        assert reset_mod.get_valid_targets("code") == [
            "discovered",
            "scored",
            "triaged",
        ]

    def test_get_valid_targets_documents(self, reset_mod):
        assert reset_mod.get_valid_targets("documents") == ["discovered", "scored"]

    def test_get_valid_targets_unknown_domain(self, reset_mod):
        assert reset_mod.get_valid_targets("nonexistent") == []

    def test_signal_discovered_spec_fields(self, reset_mod):
        spec = reset_mod.SIGNAL_RESET_SPECS["discovered"]
        assert spec.label == "FacilitySignal"
        assert spec.target_status == "discovered"
        assert "enriched" in spec.source_statuses
        assert "checked" in spec.source_statuses
        assert "underspecified" in spec.source_statuses
        assert "description" in spec.clear_fields
        assert "embedding" in spec.clear_fields
        assert "checked" in spec.clear_fields
        assert spec.facility_via_rel is False

    def test_signal_discovered_clears_embed_failed_at(self, reset_mod):
        spec = reset_mod.SIGNAL_RESET_SPECS["discovered"]
        assert "embedding" in spec.clear_fields
        assert "embed_failed_at" in spec.clear_fields

    def test_signal_enriched_spec_has_post_cypher(self, reset_mod):
        spec = reset_mod.SIGNAL_RESET_SPECS["enriched"]
        assert spec.post_cypher is not None
        assert "CHECKED_WITH" in spec.post_cypher

    def test_path_specs_use_facility_rel(self, reset_mod):
        for name, spec in reset_mod.PATH_RESET_SPECS.items():
            assert spec.facility_via_rel is True, f"paths/{name} should use rel"
            assert spec.label == "FacilityPath"

    def test_path_scanned_reset_clears_expanded_at(self, reset_mod):
        """Re-listing a root requires clearing its expansion stamp."""
        spec = reset_mod.PATH_RESET_SPECS["scanned"]
        assert "expanded_at" in spec.clear_fields

    def test_path_triaged_reset_clears_expanded_at(self, reset_mod):
        spec = reset_mod.PATH_RESET_SPECS["triaged"]
        assert "expanded_at" in spec.clear_fields

    def test_path_scanned_reset_keeps_discovered_out_of_source_statuses(
        self, reset_mod
    ):
        """The shared spec must not list ``discovered``: an unscoped reset would
        otherwise push every never-scanned discovered row past enumeration."""
        spec = reset_mod.PATH_RESET_SPECS["scanned"]
        assert "discovered" not in spec.source_statuses
        assert "triaged" in spec.source_statuses
        assert "scored" in spec.source_statuses
        # Clearing the triage composite is what makes the row claimable again.
        assert "triage_composite" in spec.clear_fields

    def test_path_scanned_reset_admits_only_stranded_discovered_rows(self, reset_mod):
        """A ``discovered`` row is admitted only when it carries a triage score,
        which distinguishes a re-seeded root from a never-scanned row."""
        spec = reset_mod.PATH_RESET_SPECS["scanned"]
        assert spec.source_filter is not None
        assert "n.status = 'discovered'" in spec.source_filter
        assert "n.triage_composite IS NOT NULL" in spec.source_filter


# ─── reset_to_status tests ──────────────────────────────────────────────


class TestResetToStatus:
    """Test the generic reset_to_status function."""

    def _mock_gc(self):
        """Create a mock GraphClient context manager."""
        mock_gc = MagicMock()
        mock_gc_ctx = MagicMock()
        mock_gc_ctx.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_ctx.__exit__ = MagicMock(return_value=False)
        return mock_gc, mock_gc_ctx

    def test_reset_signals_discovered(self, reset_mod):
        """Reset enriched signals to discovered."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 42}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.SIGNAL_RESET_SPECS["discovered"]
            count = reset_mod.reset_to_status(spec, "jet")

        assert count == 42
        mock_gc.query.assert_called_once()
        query = mock_gc.query.call_args[0][0]
        assert "FacilitySignal" in query
        assert "facility_id" in query
        assert "n.status = $target_status" in query
        assert "n.description = null" in query
        assert "n.embedding = null" in query
        assert "n.embed_failed_at = null" in query

    def test_reset_paths_triaged(self, reset_mod):
        """Reset scored paths to triaged (via relationship)."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 10}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.PATH_RESET_SPECS["triaged"]
            count = reset_mod.reset_to_status(spec, "tcv")

        assert count == 10
        query = mock_gc.query.call_args[0][0]
        assert "AT_FACILITY" in query
        assert "Facility" in query

    def test_reset_with_extra_filter(self, reset_mod):
        """Reset with extra filter passes additional params."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 5}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.SIGNAL_RESET_SPECS["discovered"]
            count = reset_mod.reset_to_status(
                spec,
                "jet",
                extra_filter="AND n.discovery_source IN $sources",
                extra_params={"sources": ["ppf"]},
            )

        assert count == 5
        call_kwargs = mock_gc.query.call_args[1]
        assert call_kwargs["sources"] == ["ppf"]

    def test_reset_empty_result(self, reset_mod):
        """Returns 0 when no results."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = []

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.WIKI_RESET_SPECS["scanned"]
            count = reset_mod.reset_to_status(spec, "jt-60sa")

        assert count == 0

    def test_post_cypher_included(self, reset_mod):
        """Specs with post_cypher include it in the query."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 3}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.SIGNAL_RESET_SPECS["enriched"]
            reset_mod.reset_to_status(spec, "jet")

        query = mock_gc.query.call_args[0][0]
        assert "CHECKED_WITH" in query
        assert "DELETE r" in query

    def test_reset_paths_clears_expanded_at_in_query(self, reset_mod):
        """The scanned reset nulls expanded_at so the root can expand again."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 12}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.PATH_RESET_SPECS["scanned"]
            reset_mod.reset_to_status(spec, "jt-60sa")

        query = mock_gc.query.call_args[0][0]
        assert "n.expanded_at = null" in query

    def test_code_discovered_reset_includes_skipped(self, reset_mod):
        """A triage-skipped code file is reset to discovered so it can be re-triaged."""
        spec = reset_mod.CODE_RESET_SPECS["discovered"]
        assert "skipped" in spec.source_statuses

    def test_reset_path_prefixes_scopes_the_query(self, reset_mod):
        """A prefix-scope reset renders the prefix clause and carries the list,
        so only rows whose path starts with a named prefix can match."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 7}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.CODE_RESET_SPECS["discovered"]
            count = reset_mod.reset_to_status(
                spec,
                "jt-60sa",
                path_prefixes=["/analysis/src/SAeqread"],
            )

        assert count == 7
        query = mock_gc.query.call_args[0][0]
        assert "n.path STARTS WITH prefix" in query
        assert mock_gc.query.call_args[1]["path_prefixes"] == ["/analysis/src/SAeqread"]

    def test_reset_without_prefixes_stays_facility_wide(self, reset_mod):
        """No prefixes keeps the facility-wide reset: no prefix clause is rendered."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 3}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.CODE_RESET_SPECS["discovered"]
            reset_mod.reset_to_status(spec, "jt-60sa")

        query = mock_gc.query.call_args[0][0]
        assert "STARTS WITH" not in query
        assert "path_prefixes" not in query
        assert "path_prefixes" not in mock_gc.query.call_args[1]

    def test_scanned_reset_renders_the_stranded_row_predicate(self, reset_mod):
        """reset_to_status ORs the spec's source_filter into the status match, so
        a discovered row with a triage score *is* reset."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 3}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.PATH_RESET_SPECS["scanned"]
            count = reset_mod.reset_to_status(spec, "jt-60sa")

        assert count == 3
        query = mock_gc.query.call_args[0][0]
        assert (
            "n.status IN $source_statuses "
            "OR (n.status = 'discovered' AND n.triage_composite IS NOT NULL)"
        ) in query

    def test_scanned_reset_excludes_never_scanned_discovered_rows(self, reset_mod):
        """The predicate requires a non-null triage_composite, so a discovered
        row with no triage score *is not* reset."""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 0}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.PATH_RESET_SPECS["scanned"]
            reset_mod.reset_to_status(spec, "jt-60sa")

        query = mock_gc.query.call_args[0][0]
        assert "n.triage_composite IS NOT NULL" in query
        assert "n.triage_composite IS NULL" not in query

    def test_reset_without_source_filter_keeps_plain_status_match(self, reset_mod):
        """A spec with no source_filter renders the bare status predicate with
        no OR branch.  (The ``triaged`` spec carries none.)"""
        mock_gc, mock_gc_ctx = self._mock_gc()
        mock_gc.query.return_value = [{"reset_count": 1}]

        with patch("imas_codex.graph.GraphClient", return_value=mock_gc_ctx):
            spec = reset_mod.PATH_RESET_SPECS["triaged"]
            reset_mod.reset_to_status(spec, "jt-60sa")

        query = mock_gc.query.call_args[0][0]
        assert "WHERE n.status IN $source_statuses" in query
        assert "$source_statuses OR" not in query

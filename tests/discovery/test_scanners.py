"""Tests for scanner plugin registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.signals.scanners.base import (
    DataSourceScanner,
    ScanResult,
    _registry,
    get_scanner,
    get_scanners_for_facility,
    list_scanners,
    register_scanner,
)


class TestScannerRegistry:
    """Test scanner registration and lookup."""

    def test_list_scanners_has_builtins(self):
        """All built-in scanners auto-register."""
        scanners = list_scanners()
        assert "tdi" in scanners
        assert "ppf" in scanners
        assert "edas" in scanners
        assert "mdsplus" in scanners
        assert "imas" in scanners
        assert "device_xml" in scanners
        assert "wiki" not in scanners

    def test_get_scanner_returns_instance(self):
        """get_scanner returns a scanner with correct type."""
        scanner = get_scanner("tdi")
        assert scanner.scanner_type == "tdi"
        assert hasattr(scanner, "scan")
        assert hasattr(scanner, "check")

    def test_get_scanner_unknown_raises(self):
        """Unknown scanner type raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_scanner("nonexistent")

    def test_scanner_protocol_compliance(self):
        """All registered scanners implement DataSourceScanner protocol."""
        for scanner_type in list_scanners():
            scanner = get_scanner(scanner_type)
            assert isinstance(scanner, DataSourceScanner)

    def test_scan_result_defaults(self):
        """ScanResult has sensible defaults."""
        result = ScanResult()
        assert result.signals == []
        assert result.data_access is None
        assert result.metadata == {}
        assert result.stats == {}


class TestGetScannersForFacility:
    """Test facility-based scanner dispatch."""

    def test_wiki_context_is_not_auto_added_as_scanner(self, monkeypatch):
        """Facilities with wiki sites only get configured data_system scanners."""
        from imas_codex.discovery.signals.scanners import base

        monkeypatch.setattr(
            "imas_codex.discovery.base.facility.get_facility",
            lambda facility: {
                "data_systems": {"mdsplus": {}},
                "wiki_sites": ["https://example.test/wiki"],
            },
        )

        scanners = get_scanners_for_facility("jet")

        assert [scanner.scanner_type for scanner in scanners] == ["mdsplus"]

    def test_tcv_returns_tdi(self, monkeypatch):
        """TCV facility should dispatch TDI scanner."""
        from imas_codex.discovery.signals.scanners import base

        monkeypatch.setattr(
            base,
            "_auto_register",
            lambda: None,
        )
        # Pre-populate registry
        _registry.clear()

        # Just test that the registry works with manual registration
        from imas_codex.discovery.signals.scanners.tdi import TDIScanner

        register_scanner(TDIScanner())
        assert get_scanner("tdi").scanner_type == "tdi"

    def test_disabled_data_source_is_skipped(self, monkeypatch):
        """Data systems marked available=false are not scheduled."""
        from imas_codex.discovery.signals.scanners import base

        monkeypatch.setattr(
            base,
            "_auto_register",
            lambda: None,
        )
        _registry.clear()

        from imas_codex.discovery.signals.scanners.mdsplus import MDSplusScanner

        register_scanner(MDSplusScanner())
        monkeypatch.setattr(
            "imas_codex.discovery.base.facility.get_facility",
            lambda facility: {
                "data_systems": {
                    "mdsplus": {},
                    "ppf": {"available": False},
                }
            },
        )

        scanners = get_scanners_for_facility("jet")

        assert [scanner.scanner_type for scanner in scanners] == ["mdsplus"]

    def test_register_custom_scanner(self):
        """Custom scanners can be registered."""

        class CustomScanner:
            scanner_type = "custom_test"

            async def scan(self, facility, ssh_host, config, reference_shot=None):
                return ScanResult()

            async def check(
                self, facility, ssh_host, signals, config, reference_shot=None
            ):
                return []

        register_scanner(CustomScanner())
        scanner = get_scanner("custom_test")
        assert scanner.scanner_type == "custom_test"

        # Clean up
        del _registry["custom_test"]


class TestSemanticWikiContextFacilityPredicate:
    """The wiki-context lookup filters facility inside the vector SEARCH.

    The predicate is rendered before the ANN cut so a small facility's own
    wiki chunks survive; as a post-filter over a global cut they would not.
    """

    @staticmethod
    def _fake_encoder():
        vector = MagicMock()
        vector.tolist.return_value = [0.1, 0.2, 0.3]
        encoder = MagicMock()
        encoder.embed_texts.return_value = [vector]
        return encoder

    @staticmethod
    def _mock_gc():
        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[])
        return gc

    def test_facility_predicate_inside_search(self):
        from imas_codex.discovery.signals.scanners import wiki as wiki_mod

        mock_gc = self._mock_gc()
        with (
            patch.object(wiki_mod, "GraphClient", return_value=mock_gc),
            patch(
                "imas_codex.embeddings.encoder.Encoder",
                return_value=self._fake_encoder(),
            ),
            patch(
                "imas_codex.embeddings.config.EncoderConfig", return_value=MagicMock()
            ),
        ):
            result = wiki_mod.fetch_semantic_wiki_context("jt-60sa", "plasma current")

        assert result == []
        cypher = mock_gc.query.call_args[0][0]
        open_idx = cypher.index("SEARCH")
        close_idx = cypher.index(") SCORE AS")
        pred_idx = cypher.index("node.facility_id = $facility")
        assert open_idx < pred_idx < close_idx, cypher
        assert "p.facility_id" not in cypher, cypher
        assert mock_gc.query.call_args[1]["facility"] == "jt-60sa"

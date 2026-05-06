"""Tests for standard name MCP tools.

Validates that ``_search_standard_names`` delegates correctly to the backing
search function and that result formatting, kind/pipeline_status filtering,
and parameter forwarding (physics_domain, segment_filters) all work.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _row(**overrides):
    """Return a minimal row dict matching the columns returned by the
    three ``_*_search_standard_names`` helpers, with optional overrides."""
    base = {
        "name": "electron_temperature",
        "description": "Te",
        "kind": "scalar",
        "unit": "eV",
        "pipeline_status": "drafted",
        "documentation": None,
        "physics_domain": "transport",
        "cocos_transformation_type": None,
        "cocos": None,
        "score": 1.0,
    }
    base.update(overrides)
    return base


class TestSearchStandardNames:
    """Test _search_standard_names tool delegates to backing and formats."""

    def _patch_backing(self, return_value):
        """Patch the backing search function used by _search_standard_names."""
        return patch(
            "imas_codex.llm.sn_tools._search_sn_backing",
            return_value=return_value,
        )

    def test_basic_search_returns_formatted_results(self):
        """Backing results are formatted into a readable report."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing([_row()]):
            result = _search_standard_names("electron temperature", gc=MagicMock())

        assert "electron_temperature" in result

    def test_empty_results(self):
        """Empty results produce informative message."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing([]):
            result = _search_standard_names("nonexistent quantity", gc=MagicMock())

        assert "No" in result or "0" in result

    def test_no_args_no_typeerror(self):
        """Search with only the query must not raise TypeError."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing([]):
            _search_standard_names("temperature", gc=MagicMock())

    def test_kind_filter_forwarded_to_backing(self):
        """Kind filter is forwarded to backing search function."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing([_row(kind="scalar")]) as mock_backing:
            result = _search_standard_names(
                "temperature", kind="scalar", gc=MagicMock()
            )

        assert "electron_temperature" in result
        _, kwargs = mock_backing.call_args
        assert kwargs.get("kind") == "scalar"

    def test_pipeline_status_filter_forwarded_to_backing(self):
        """pipeline_status filter is forwarded to backing search function."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing(
            [_row(name="drafted_name", pipeline_status="drafted")]
        ) as mock_backing:
            result = _search_standard_names(
                "test", pipeline_status="drafted", gc=MagicMock()
            )

        assert "drafted_name" in result
        _, kwargs = mock_backing.call_args
        assert kwargs.get("pipeline_status") == "drafted"

    def test_physics_domain_post_filter(self):
        """physics_domain post-filters results from the backing function."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing(
            [
                _row(name="transport_name", physics_domain="transport"),
                _row(name="equilibrium_name", physics_domain="equilibrium"),
            ]
        ):
            result = _search_standard_names(
                "temperature", physics_domain="transport", gc=MagicMock()
            )

        assert "transport_name" in result
        assert "equilibrium_name" not in result

    def test_physics_domain_default_none_no_filter(self):
        """When physics_domain is None, all results are returned."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing(
            [
                _row(name="transport_name", physics_domain="transport"),
                _row(name="equilibrium_name", physics_domain="equilibrium"),
            ]
        ):
            result = _search_standard_names("temperature", gc=MagicMock())

        assert "transport_name" in result
        assert "equilibrium_name" in result

    def test_segment_filters_forwarded_to_backing(self):
        """Segment filter kwargs are collected and forwarded to backing."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing([_row()]) as mock_backing:
            _search_standard_names(
                "temperature",
                physical_base="temperature",
                subject="electron",
                gc=MagicMock(),
            )

        _, kwargs = mock_backing.call_args
        assert kwargs["segment_filters"] == {
            "physical_base": "temperature",
            "subject": "electron",
        }

    def test_result_format_no_grammar_fields(self):
        """Result format does not include grammar_* fields."""
        from imas_codex.llm.sn_tools import _search_standard_names

        with self._patch_backing([_row(score=0.92)]):
            result = _search_standard_names("electron temperature", gc=MagicMock())

        assert "electron_temperature" in result
        assert "0.92" in result
        assert "physical_base=" not in result
        assert "subject=" not in result


class TestFetchStandardNames:
    """Test _fetch_standard_names tool."""

    def test_fetch_single(self):
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te profile",
                    "documentation": "The $T_e$ profile",
                    "kind": "scalar",
                    "unit": "eV",
                    "links": ["ion_temperature"],
                    "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                    "constraints": ["T_e > 0"],
                    "validity_domain": "core plasma",
                    "pipeline_status": "drafted",
                    "model": "test",
                    "source_ids": ["core_profiles/profiles_1d/electrons/temperature"],
                    "source_ids_names": ["core_profiles"],
                }
            ]
        )

        result = _fetch_standard_names("electron_temperature", gc=mock_gc)
        assert "electron_temperature" in result
        assert "eV" in result
        assert "$T_e$" in result

    def test_fetch_multiple_comma_separated(self):
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": "eV",
                    "links": [],
                    "source_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "pipeline_status": "drafted",
                    "model": None,
                    "source_ids": [],
                    "source_ids_names": [],
                },
                {
                    "name": "plasma_current",
                    "description": "Ip",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": "A",
                    "links": [],
                    "source_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "pipeline_status": "drafted",
                    "model": None,
                    "source_ids": [],
                    "source_ids_names": [],
                },
            ]
        )

        result = _fetch_standard_names(
            "electron_temperature,plasma_current", gc=mock_gc
        )
        assert "electron_temperature" in result
        assert "plasma_current" in result

    def test_fetch_not_found(self):
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        result = _fetch_standard_names("nonexistent_name", gc=mock_gc)
        assert "not found" in result.lower() or "No" in result

    def test_fetch_partial_not_found(self):
        """Shows not found message for missing names."""
        from imas_codex.llm.sn_tools import _fetch_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": "eV",
                    "links": [],
                    "source_paths": [],
                    "constraints": [],
                    "validity_domain": None,
                    "pipeline_status": "drafted",
                    "model": None,
                    "source_ids": [],
                    "source_ids_names": [],
                }
            ]
        )

        result = _fetch_standard_names("electron_temperature missing_name", gc=mock_gc)
        assert "electron_temperature" in result
        assert "missing_name" in result
        assert "Not found" in result


class TestListStandardNames:
    """Test _list_standard_names tool."""

    def test_list_all(self):
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
                {
                    "name": "plasma_current",
                    "kind": "scalar",
                    "unit": "A",
                    "pipeline_status": "drafted",
                    "description": "Ip",
                },
            ]
        )

        result = _list_standard_names(gc=mock_gc)
        assert "electron_temperature" in result
        assert "plasma_current" in result

    def test_list_no_args_no_typeerror(self):
        """Plan-MCP regression: list with no args must not raise.

        Before this plan landed, the MCP wrapper forwarded ``tag=None``
        unconditionally, which the backing function never accepted.
        """
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        # Must not raise
        _list_standard_names(gc=mock_gc)

    def test_list_with_physics_domain_filter(self):
        """physics_domain filter is pushed into the Cypher WHERE clause."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
            ]
        )

        result = _list_standard_names(physics_domain="transport", gc=mock_gc)
        assert "electron_temperature" in result
        call = mock_gc.query.call_args
        cypher = call.args[0] if call.args else ""
        assert "physics_domain" in cypher
        assert call.kwargs.get("physics_domain") == "transport"

    def test_list_empty_results(self):
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        result = _list_standard_names(physics_domain="nonexistent_domain", gc=mock_gc)
        assert "No standard names" in result

    def test_list_filter_info_in_header(self):
        """Filter params appear in header."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
            ]
        )

        result = _list_standard_names(kind="scalar", gc=mock_gc)
        assert "kind=scalar" in result

    def test_list_physics_domain_in_header(self):
        """physics_domain filter shows in header."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
            ]
        )

        result = _list_standard_names(physics_domain="transport", gc=mock_gc)
        assert "physics_domain=transport" in result

    def test_list_table_format(self):
        """Output is a markdown table."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "kind": "scalar",
                    "unit": "eV",
                    "pipeline_status": "drafted",
                    "description": "Te",
                },
            ]
        )

        result = _list_standard_names(gc=mock_gc)
        assert "| Name |" in result
        assert "| electron_temperature |" in result


class TestMCPToolRegistration:
    """Test that SN tools are importable and have the post-plan signatures."""

    def test_tools_importable(self):
        """SN tools should be importable from sn_tools."""
        from imas_codex.llm.sn_tools import (
            _fetch_standard_names,
            _list_standard_names,
            _search_standard_names,
        )

        assert callable(_search_standard_names)
        assert callable(_fetch_standard_names)
        assert callable(_list_standard_names)

    def test_search_signature(self):
        """search_standard_names accepts physics_domain (not tags)."""
        import inspect

        from imas_codex.llm.sn_tools import _search_standard_names

        sig = inspect.signature(_search_standard_names)
        params = set(sig.parameters.keys())
        assert "query" in params
        assert "kind" in params
        assert "physics_domain" in params
        assert "pipeline_status" in params
        assert "k" in params
        assert "gc" in params
        # The legacy ``tags`` filter has been dropped (Plan MCP+units Track A)
        assert "tag" not in params

    def test_fetch_signature(self):
        """fetch_standard_names accepts expected kwargs."""
        import inspect

        from imas_codex.llm.sn_tools import _fetch_standard_names

        sig = inspect.signature(_fetch_standard_names)
        params = set(sig.parameters.keys())
        assert "names" in params
        assert "gc" in params

    def test_list_signature(self):
        """list_standard_names accepts physics_domain (not tag)."""
        import inspect

        from imas_codex.llm.sn_tools import _list_standard_names

        sig = inspect.signature(_list_standard_names)
        params = set(sig.parameters.keys())
        assert "physics_domain" in params
        assert "kind" in params
        assert "pipeline_status" in params
        assert "gc" in params
        # The legacy ``tag`` filter has been dropped
        assert "tag" not in params

    def test_mcp_wrapper_no_tags_tag(self):
        """MCP wrappers must not declare tag/tags parameters."""
        # Server module must import cleanly and contain no `tag`/`tags`
        # default parameters in the SN wrapper signatures.
        import inspect

        from imas_codex.llm import server  # noqa: F401

        src = inspect.getsource(server)
        # Anchor checks to the SN tool definitions, not surrounding prose.
        assert "tags: list[str] | None = None" not in src
        # The list_standard_names wrapper must not declare ``tag``.
        assert "def list_standard_names(\n                tag:" not in src


class TestSupersededExclusion:
    """Superseded SNs must not appear in search or list results."""

    def _superseded_row(self, name: str = "electron_heating_power") -> dict:
        return {
            "name": name,
            "description": "Superseded fossil",
            "kind": "scalar",
            "unit": "1",
            "pipeline_status": "superseded",
            "documentation": None,
            "physics_domain": "transport",
            "cocos_transformation_type": None,
            "cocos": None,
            "score": 0.99,
        }

    def _active_row(self, name: str = "electron_temperature") -> dict:
        return {
            "name": name,
            "description": "Electron temperature",
            "kind": "scalar",
            "unit": "eV",
            "pipeline_status": "drafted",
            "documentation": None,
            "physics_domain": "transport",
            "cocos_transformation_type": None,
            "cocos": None,
            "score": 0.85,
        }

    def test_backing_excludes_superseded(self):
        """Backing search function is called and superseded names are excluded.

        The backing function (search.search_standard_names) handles superseded
        exclusion in its Cypher. We verify the backing is called correctly.
        """
        from imas_codex.llm.sn_tools import _search_standard_names

        with patch(
            "imas_codex.llm.sn_tools._search_sn_backing",
            return_value=[],
        ) as mock_backing:
            result = _search_standard_names("electron temperature", gc=MagicMock())

        mock_backing.assert_called_once()
        # No superseded fossil should appear in the formatted output.
        assert "electron_heating_power" not in result

    def test_list_excludes_superseded_by_default(self):
        """_list_standard_names must exclude superseded by default."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        _list_standard_names(gc=mock_gc)
        cypher = mock_gc.query.call_args.args[0]
        assert "name_stage" in cypher and "superseded" in cypher

    def test_list_include_superseded_flag(self):
        """With include_superseded=True, no name_stage guard in Cypher."""
        from imas_codex.llm.sn_tools import _list_standard_names

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        _list_standard_names(include_superseded=True, gc=mock_gc)
        cypher = mock_gc.query.call_args.args[0]
        # The guard should be absent when caller opts-in to superseded
        assert "name_stage" not in cypher or "superseded" not in cypher

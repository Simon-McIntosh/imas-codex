"""Contract tests for ISN grammar API stability."""

from __future__ import annotations


def test_grammar_context_contract():
    """Verify get_grammar_context() returns all keys codex depends on."""
    from imas_standard_names.grammar.context import get_grammar_context

    ctx = get_grammar_context()
    required = {
        "canonical_pattern",
        "segment_order",
        "template_rules",
        "vocabulary_sections",
        "segment_descriptions",
        "naming_guidance",
        "kind_definitions",
        "anti_patterns",
        "exclusive_pairs",
        "field_guidance",
        "applicability",
        "quick_start",
        "common_patterns",
        "critical_distinctions",
        "vocabulary_usage_stats",
        "base_requirements",
        "type_specific_requirements",
        "documentation_guidance",
    }
    assert required <= set(ctx.keys()), f"Missing keys: {required - set(ctx.keys())}"


def test_grammar_context_types():
    """Verify key types from get_grammar_context()."""
    from imas_standard_names.grammar.context import get_grammar_context

    ctx = get_grammar_context()
    assert isinstance(ctx["canonical_pattern"], str)
    assert isinstance(ctx["segment_order"], str)
    assert isinstance(ctx["template_rules"], str)
    assert isinstance(ctx["vocabulary_sections"], list)
    assert isinstance(ctx["segment_descriptions"], dict)
    assert isinstance(ctx["exclusive_pairs"], list)


def test_standard_name_entry_import():
    """Verify StandardNameEntry is importable from models."""
    from imas_standard_names.models import StandardNameEntry, create_standard_name_entry

    assert StandardNameEntry is not None
    assert callable(create_standard_name_entry)


def test_build_compose_context_has_isn_keys():
    """Verify compose context includes ISN-provided keys."""
    from imas_codex.standard_names.context import (
        build_compose_context,
        clear_context_cache,
    )

    clear_context_cache()
    ctx = build_compose_context()
    # Core grammar keys
    assert "canonical_pattern" in ctx
    assert "segment_order" in ctx
    assert "vocabulary_sections" in ctx
    assert "segment_descriptions" in ctx
    # New ISN-provided keys
    assert "naming_guidance" in ctx
    assert "kind_definitions" in ctx
    assert "anti_patterns" in ctx
    # Additional ISN keys
    assert "quick_start" in ctx
    assert "common_patterns" in ctx
    assert "critical_distinctions" in ctx
    assert "vocabulary_usage_stats" in ctx
    assert "base_requirements" in ctx
    assert "type_specific_requirements" in ctx
    assert "documentation_guidance" in ctx
    # Codex-specific keys still present
    assert "examples" in ctx
    assert "tokamak_ranges" in ctx
    assert "field_guidance" in ctx
    assert "applicability" in ctx
    assert "exclusive_pairs" in ctx
    clear_context_cache()


# ── Grammar segment extraction tests ──


def test_extract_segments_physical_vector_component():
    """Verify extraction of component + physical_base from a vector projection."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("toroidal_component_of_magnetic_field")
    assert result["grammar_physical_base"] == "magnetic_field"
    assert result["grammar_component"] == "toroidal"
    assert result["grammar_coordinate"] is None
    assert result["grammar_geometric_base"] is None
    assert result["grammar_parse_version"] is not None


def test_extract_segments_scalar_quantity():
    """Verify extraction of physical_base for a plain scalar."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("pressure")
    assert result["grammar_physical_base"] == "pressure"
    assert result["grammar_component"] is None
    assert result["grammar_coordinate"] is None


def test_extract_segments_unparseable_name():
    """Unparseable names get version but null segment fields."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("nonexistent_gibberish_xyzzy")
    assert result["grammar_parse_version"] is not None
    assert result["grammar_physical_base"] is None
    assert result["grammar_component"] is None
    assert result["validation_diagnostics_json"] == "[]"


def test_extract_segments_parseable_compound():
    """Parseable compound names extract subject + physical_base via Pydantic enrichment."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("electron_temperature")
    assert result["grammar_physical_base"] == "temperature"
    assert result["grammar_subject"] == "electron"
    assert result["grammar_component"] is None


def test_extract_segments_all_keys_present():
    """All grammar_* keys must be present even on parse failure."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("nonexistent_gibberish_xyzzy")
    expected_keys = {
        "grammar_parse_version",
        "validation_diagnostics_json",
        "grammar_physical_base",
        "grammar_geometric_base",
        "grammar_subject",
        "grammar_component",
        "grammar_coordinate",
        "grammar_transformation",
        "grammar_position",
        "grammar_process",
        "grammar_device",
        "grammar_region",
    }
    assert expected_keys <= set(result.keys())


# ── Grammar field persistence tests (P1) ──


def test_write_standard_names_persists_grammar_segments():
    """write_standard_names MERGE Cypher must SET all grammar_* segment fields."""
    from unittest.mock import MagicMock, patch

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_standard_names

        write_standard_names(
            [
                {
                    "id": "electron_temperature",
                    "source_types": ["dd"],
                    "source_id": "core_profiles/profiles_1d/electrons/temperature",
                }
            ]
        )

    # Find the MERGE StandardName Cypher
    merge_cypher = None
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if "MERGE (sn:StandardName" in cypher:
            merge_cypher = cypher
            break
    assert merge_cypher is not None, "No MERGE StandardName query found"

    # All grammar segment fields must appear in the SET clause with coalesce
    grammar_fields = [
        "grammar_physical_base",
        "grammar_geometric_base",
        "grammar_subject",
        "grammar_component",
        "grammar_coordinate",
        "grammar_transformation",
        "grammar_position",
        "grammar_process",
        "grammar_device",
        "grammar_region",
    ]
    for field in grammar_fields:
        assert f"sn.{field}" in merge_cypher, f"MERGE Cypher must SET sn.{field}"
        assert f"coalesce(b.{field}" in merge_cypher, (
            f"MERGE Cypher must use coalesce for {field}"
        )


def test_write_standard_names_batch_includes_grammar_segments():
    """write_standard_names batch dict must include grammar segment values."""
    from unittest.mock import MagicMock, patch

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_standard_names

        write_standard_names(
            [
                {
                    "id": "electron_temperature",
                    "source_types": ["dd"],
                    "source_id": "core_profiles/profiles_1d/electrons/temperature",
                }
            ]
        )

    # Find batch from MERGE call
    batch = None
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if "MERGE (sn:StandardName" in cypher:
            batch = c[1]["batch"]
            break
    assert batch is not None
    entry = batch[0]

    # electron_temperature → physical_base=temperature, subject=electron
    assert entry["grammar_physical_base"] == "temperature", (
        "grammar_physical_base should be 'temperature' for electron_temperature"
    )
    assert entry["grammar_subject"] == "electron", (
        "grammar_subject should be 'electron' for electron_temperature"
    )

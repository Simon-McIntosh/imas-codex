"""Contract tests for ISN grammar API stability."""

from __future__ import annotations

import pytest


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

    result = _parse_grammar("toroidal_magnetic_field")
    assert result["physical_base"] == "magnetic_field"
    assert result["component"] == "toroidal"
    assert result["coordinate"] is None
    assert result["geometric_base"] is None
    assert result["grammar_parse_version"] is not None


def test_extract_segments_bare_generic_base_is_fallback():
    """A bare generic base (e.g. 'pressure') is model-invalid → fallback.

    The ISN pydantic model rejects generic physical_base tokens without a
    subject; since the model is the single accept/reject authority, the
    persist path records fallback with all segment columns None — exactly
    like ``_write_grammar_decomposition``.
    """
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("pressure")
    assert result["grammar_parse_fallback"] is True
    assert result["physical_base"] is None
    assert result["component"] is None
    assert result["coordinate"] is None


def test_extract_segments_unparseable_name():
    """Unparseable names get version + fallback but null segment fields."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("nonexistent_gibberish_xyzzy")
    assert result["grammar_parse_version"] is not None
    assert result["grammar_parse_fallback"] is True
    assert result["physical_base"] is None
    assert result["component"] is None
    assert result["validation_diagnostics_json"] == "[]"


def test_extract_segments_parseable_compound():
    """Parseable compound names extract subject + physical_base via Pydantic enrichment."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("electron_temperature")
    assert result["physical_base"] == "temperature"
    assert result["subject"] == "electron"
    assert result["component"] is None


def test_extract_segments_all_keys_present():
    """All bare segment keys must be present even on parse failure."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("nonexistent_gibberish_xyzzy")
    expected_keys = {
        "grammar_parse_version",
        "validation_diagnostics_json",
        "grammar_parse_fallback",
        "physical_base",
        "geometric_base",
        "subject",
        "aggregation",
        "orbit",
        "population",
        "component",
        "coordinate",
        "transformation",
        "position",
        "process",
        "device",
        "region",
        "object",
        "geometry",
    }
    assert expected_keys <= set(result.keys())


def test_extract_segments_new_single_token_modifiers():
    """aggregation / orbit / population are extracted from the pydantic model."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    result = _parse_grammar("total_trapped_fast_ion_energy")
    assert result["aggregation"] == "total"
    assert result["orbit"] == "trapped"
    assert result["population"] == "fast"
    assert result["subject"] == "ion"
    assert result["physical_base"] == "energy"


# ── Persist / decomposition parser parity ──


def _decomposition_write(name: str) -> tuple[bool, dict | None]:
    """Run _write_grammar_decomposition against a mock graph.

    Returns ``(fallback, columns)`` where *columns* is the segment-value
    dict passed to the column-write Cypher (None on fallback).
    """
    from unittest.mock import MagicMock

    from imas_codex.standard_names.graph_ops import (
        _GRAMMAR_SEGMENT_COLUMNS,
        _write_grammar_decomposition,
    )

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])
    _write_grammar_decomposition(mock_gc, [name])
    for call in mock_gc.query.call_args_list:
        cypher = call[0][0]
        if "grammar_parse_fallback = true" in cypher:
            return True, None
        if "grammar_parse_fallback = false" in cypher:
            return False, {seg: call[1].get(seg) for seg in _GRAMMAR_SEGMENT_COLUMNS}
    raise AssertionError(f"no column-write query issued for {name!r}")


@pytest.mark.parametrize(
    ("name", "expect_fallback"),
    [
        ("total_trapped_fast_ion_energy", False),
        ("fast_thermal_ion_density", True),  # same-segment population stacking
        ("pressure", True),  # bare generic base — model-invalid
        ("nonexistent_gibberish_xyzzy", True),  # unknown base token
    ],
)
def test_persist_and_decomposition_parity(name: str, expect_fallback: bool):
    """The persist path (_parse_grammar) and the decomposition writer
    (_write_grammar_decomposition) share the same pydantic accept/reject
    authority: for ANY name string the fallback flags agree and, on
    success, the segment columns equal the pydantic model fields.
    """
    from imas_standard_names.grammar import parse_standard_name

    from imas_codex.standard_names.graph_ops import (
        _GRAMMAR_SEGMENT_COLUMNS,
        _parse_grammar,
        _segments_from_model,
    )

    pg = _parse_grammar(name)
    decomp_fallback, decomp_cols = _decomposition_write(name)

    assert pg["grammar_parse_fallback"] is expect_fallback
    assert decomp_fallback is expect_fallback

    persist_cols = {seg: pg[seg] for seg in _GRAMMAR_SEGMENT_COLUMNS}
    if expect_fallback:
        assert all(v is None for v in persist_cols.values()), (
            f"fallback must clear all persist columns for {name!r}"
        )
    else:
        assert persist_cols == decomp_cols, (
            f"persist and decomposition columns diverge for {name!r}"
        )
        assert persist_cols == _segments_from_model(parse_standard_name(name)), (
            f"columns must equal the pydantic model fields for {name!r}"
        )


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
        "physical_base",
        "geometric_base",
        "subject",
        "aggregation",
        "orbit",
        "population",
        "component",
        "coordinate",
        "transformation",
        "position",
        "process",
        "device",
        "region",
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
    assert entry["physical_base"] == "temperature", (
        "physical_base should be 'temperature' for electron_temperature"
    )
    assert entry["subject"] == "electron", (
        "subject should be 'electron' for electron_temperature"
    )

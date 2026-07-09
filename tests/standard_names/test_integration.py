"""Integration tests for embedding coverage and coalesce safety.

Verifies that:
1. Embedding fields are never accidentally erased by write_standard_names.
2. All optional fields in write_standard_names use coalesce so that a
   None value in the batch never overwrites existing graph data.
3. created_at is preserved across rewrites.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

imas_sn = pytest.importorskip("imas_standard_names")

# =============================================================================
# Helpers
# =============================================================================


def _call_write(names: list[dict], mock_gc: MagicMock) -> int:
    """Call write_standard_names with a mocked GraphClient."""
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_standard_names

        return write_standard_names(names)


def _merge_cypher(mock_gc: MagicMock) -> str:
    """Return the Cypher string from the first MERGE StandardName query call."""
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if "MERGE (sn:StandardName" in cypher:
            return cypher
    raise AssertionError("No MERGE StandardName query found in calls")


def _merge_batch(mock_gc: MagicMock) -> list[dict]:
    """Return the batch parameter from the first MERGE StandardName query call."""
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if "MERGE (sn:StandardName" in cypher:
            return c[1]["batch"]
    raise AssertionError("No MERGE StandardName query found in calls")


# =============================================================================
# Part 1: Embedding Coverage
# =============================================================================


class TestEmbeddingCoverage:
    """Verify that embedding vectors are never accidentally erased."""

    def test_write_preserves_existing_embedding(self) -> None:
        """write_standard_names does not touch the embedding property at all.

        A StandardName that already has embedding=[0.1, 0.2, 0.3] and
        embedded_at set must be unchanged after write_standard_names is called.
        The Cypher must not reference the embedding property (since the function
        only manages metadata, not embeddings).
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Electron temperature",
                # No embedding field — write should not touch it
            }
        ]
        _call_write(names, mock_gc)

        cypher = _merge_cypher(mock_gc)

        # The MERGE SET clause must NOT set sn.embedding unconditionally
        # (any mention would risk overwriting it with null)
        assert "sn.embedding = b.embedding" not in cypher, (
            "write_standard_names must not set sn.embedding from batch param"
        )

    def test_embedding_field_not_in_write_batch(self) -> None:
        """write_standard_names must never clobber existing embeddings.

        The batch dict MAY contain an ``embedding`` key (possibly None)
        because the write function now uses ``coalesce(b.embedding,
        sn.embedding)`` in Cypher — a None value preserves the existing
        graph embedding rather than overwriting it.  This test verifies
        the coalesce pattern, which is the true semantic guarantee.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Electron temperature",
                "kind": "scalar",
                "unit": "eV",
                "name_stage": "drafted",
            }
        ]
        _call_write(names, mock_gc)

        cypher = _merge_cypher(mock_gc)
        assert "coalesce(b.embedding, sn.embedding)" in cypher, (
            "write_standard_names Cypher must use "
            "coalesce(b.embedding, sn.embedding) to preserve existing "
            "embeddings"
        )
        assert "coalesce(b.embedded_at, sn.embedded_at)" in cypher, (
            "write_standard_names Cypher must use "
            "coalesce(b.embedded_at, sn.embedded_at)"
        )


# =============================================================================
# Part 2: Coalesce Safety
# =============================================================================


class TestCoalesceSafety:
    """Verify that write_standard_names uses coalesce for all optional fields.

    When a field is None in the batch, coalesce(None, sn.field) = sn.field,
    so an sn-build re-run cannot accidentally erase data that was set by
    an earlier catalog import.
    """

    # Plan 38 W4a removed individual grammar_* slots (grammar_physical_base,
    # grammar_subject, grammar_component, etc.) from the StandardName schema.
    # grammar_parse_version is the only grammar field written via coalesce now.
    _COALESCE_FIELDS = [
        ("documentation", "b.documentation, sn.documentation"),
        ("kind", "b.kind, sn.kind"),
        ("links", "b.links, sn.links"),
        ("source_paths", "b.source_paths, sn.source_paths"),
        ("validity_domain", "b.validity_domain, sn.validity_domain"),
        ("constraints", "b.constraints, sn.constraints"),
        ("grammar_parse_version", "b.grammar_parse_version, sn.grammar_parse_version"),
    ]

    def test_build_does_not_erase_imported_data(self) -> None:
        """All optional fields in the MERGE SET must use coalesce(b.field, sn.field).

        This protects against a scenario where:
          1. catalog import sets name_stage='accepted', documentation, etc.
          2. sn-build re-runs write_standard_names with those fields = None
          3. Without coalesce, the re-run would null-out the imported values.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        # Simulate a minimal sn-build write — only id and source_types provided
        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Electron temperature",
                # name_stage, documentation, kind, tags, etc. all absent/None
            }
        ]
        _call_write(names, mock_gc)

        cypher = _merge_cypher(mock_gc)

        for field_name, coalesce_args in self._COALESCE_FIELDS:
            assert f"coalesce({coalesce_args})" in cypher, (
                f"Field '{field_name}' must use coalesce({coalesce_args}) in "
                "write_standard_names Cypher to preserve existing graph data"
            )

    def test_build_with_none_fields_preserves_graph(self) -> None:
        """Batch dicts must include None for absent optional fields (not omit them).

        The coalesce(b.field, sn.field) pattern requires that b.field is
        present in the batch parameter (as None, not missing) so that Cypher
        can evaluate the coalesce.  If the key were absent from the dict,
        Neo4j would raise an error or behave unpredictably.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        # Entry with almost all optional fields missing
        names = [
            {
                "id": "plasma_current",
                "source_types": ["dd"],
                "source_id": "magnetics/method/0/ip",
            }
        ]
        _call_write(names, mock_gc)

        batch = _merge_batch(mock_gc)
        assert len(batch) == 1
        item = batch[0]

        # All optional fields must appear in the batch (value may be None).
        # Plan 38 W4a dropped individual grammar_* slots (grammar_physical_base,
        # grammar_subject, etc.) from the schema; they are no longer written.
        # The grammar_* fields now written are grammar_parse_version (version
        # string) and validation_diagnostics_json, both computed by
        # _parse_grammar.
        # physics_domain is NOT in the main MERGE batch — it's handled
        # separately in the promote-on-higher-rank block that reads existing
        # values, computes promotion in Python, and writes back via a
        # dedicated follow-up query.
        required_keys = {
            "id",
            "source_types",
            "description",
            "documentation",
            "kind",
            "links",
            "source_paths",
            "validity_domain",
            "constraints",
            "unit",
            "cocos_transformation_type",
            "cocos",
            "dd_version",
            "model",
            "generated_at",
            "review_tier",
            "vocab_gap_detail",
            "validation_issues",
            "validation_layer_summary",
            "validation_status",
            "link_status",
            "review_input_hash",
            "embedding",
            "embedded_at",
            "grammar_parse_version",
            "validation_diagnostics_json",
        }
        missing = required_keys - set(item.keys())
        assert not missing, (
            f"Batch item is missing keys: {missing}. "
            "All optional fields must be present (even as None) for coalesce to work."
        )

        # Fields absent from source must be None (not some unexpected value).
        # grammar_parse_version and validation_diagnostics_json are auto-computed
        # by _parse_grammar; link_status is derived from the links field.
        auto_computed = {
            "id",
            "source_types",
            "link_status",
            "grammar_parse_version",
            "validation_diagnostics_json",
        }
        for key in required_keys - auto_computed:
            assert item[key] is None, (
                f"Batch key '{key}' should be None when not supplied, got {item[key]!r}"
            )

    def test_created_at_preserved_on_rewrite(self) -> None:
        """created_at must use coalesce(sn.created_at, datetime()) — not coalesce(b.created_at, ...).

        This pattern sets created_at on first write and then leaves it
        unchanged on all subsequent writes, so the node retains its
        original creation timestamp.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
            }
        ]
        _call_write(names, mock_gc)

        cypher = _merge_cypher(mock_gc)

        # Pattern: sn.created_at is preserved on re-writes, set only on first
        assert "coalesce(sn.created_at, datetime())" in cypher, (
            "write_standard_names must use coalesce(sn.created_at, datetime()) "
            "to preserve the original creation timestamp across rewrites"
        )
        # Must not blindly set created_at from the batch
        assert "b.created_at" not in cypher, (
            "write_standard_names must not set created_at from batch param"
        )


# =============================================================================
# Full pipeline: write_standard_names field coverage
# =============================================================================

# Rich sample data shared across pipeline tests
_RICH_SN_RECORD = {
    "id": "electron_temperature",
    "source_types": ["dd"],
    "source_id": "core_profiles/profiles_1d/electrons/temperature",
    "ids_name": "core_profiles",
    "description": "Electron temperature in the core plasma",
    "documentation": (
        "The electron temperature $T_e$ is measured via Thomson scattering. "
        "It is a key parameter for transport modelling."
    ),
    "kind": "scalar",
    "unit": "eV",
    "links": ["name:ion_temperature"],
    "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "validity_domain": "core plasma",
    "constraints": ["T_e > 0"],
    "physical_base": "temperature",
    "subject": "electron",
    "model": "gpt-4o",
    "name_stage": "drafted",
}


class TestWriteStandardNames:
    """write_standard_names field coverage.

    All graph operations are mocked — no live Neo4j required.
    """

    def test_write_standard_names_called_with_all_fields(self) -> None:
        """write_standard_names receives all populated fields without losing any."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        captured_calls: list = []

        def capture_query(cypher, **kwargs):
            if "batch" in kwargs:
                captured_calls.append({"cypher": cypher, "batch": kwargs["batch"]})
            return None

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(side_effect=capture_query)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_gc)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.protection._fetch_catalog_edit_names",
                return_value=set(),
            ),
        ):
            MockGC.return_value = mock_ctx
            write_standard_names([_RICH_SN_RECORD])

        # Find the MERGE StandardName batch (not the conflict check batch)
        merge_batch = None
        for call_info in captured_calls:
            if "MERGE (sn:StandardName" in call_info["cypher"]:
                merge_batch = call_info["batch"]
                break
        assert merge_batch is not None, "MERGE StandardName query not found"
        node = merge_batch[0]
        assert node["id"] == "electron_temperature"
        assert node["description"] == "Electron temperature in the core plasma"
        assert node["kind"] == "scalar"
        assert node["unit"] == "eV"
        assert node["constraints"] == ["T_e > 0"]
        assert node["validity_domain"] == "core plasma"
        assert "name_stage" not in node  # stage owned by _finalize_generated_name_stage
        assert node["model"] == "gpt-4o"


# =============================================================================
# Part 7: LLM Cost Persistence (compose unified with review/enrich)
# =============================================================================


class TestLlmCostPersistence:
    """Verify write_standard_names persists llm_cost/tokens/model fields."""

    def test_cost_fields_written_to_merge_batch(self) -> None:
        """Cost and token fields from the compose batch must appear in the
        MERGE batch payload so SUM(sn.llm_cost) aggregates graph-wide spend.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "unit": "eV",
                "llm_cost": 0.0412,
                "llm_model": "openrouter/anthropic/claude-sonnet-4.6",
                "llm_service": "standard-names",
                "llm_at": "2026-04-23T21:16:05.937268+00:00",
                "llm_tokens_in": 28805,
                "llm_tokens_out": 275,
                "llm_tokens_cached_read": 28805,
                "llm_tokens_cached_write": 0,
            }
        ]
        _call_write(names, mock_gc)
        batch = _merge_batch(mock_gc)
        assert batch[0]["llm_cost"] == pytest.approx(0.0412)
        assert batch[0]["llm_model"] == "openrouter/anthropic/claude-sonnet-4.6"
        assert batch[0]["llm_tokens_in"] == 28805
        assert batch[0]["llm_tokens_out"] == 275
        assert batch[0]["llm_tokens_cached_read"] == 28805

    def test_merge_cypher_sets_cost_fields(self) -> None:
        """The MERGE statement must include SET clauses for every cost field
        so the graph schema's llm_* properties are populated by compose.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        names = [
            {
                "id": "plasma_current",
                "source_types": ["dd"],
                "source_id": "equilibrium/time_slice/global_quantities/ip",
                "unit": "A",
                "llm_cost": 0.05,
            }
        ]
        _call_write(names, mock_gc)
        cypher = _merge_cypher(mock_gc)
        for field in (
            "sn.llm_cost",
            "sn.llm_model",
            "sn.llm_service",
            "sn.llm_at",
            "sn.llm_tokens_in",
            "sn.llm_tokens_out",
            "sn.llm_tokens_cached_read",
            "sn.llm_tokens_cached_write",
        ):
            assert field in cypher, f"MERGE missing SET clause for {field}"

    def test_cost_fields_coalesced_not_overwritten(self) -> None:
        """Cost fields must accumulate (not overwrite) to preserve previous LLM costs."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        names = [
            {
                "id": "electron_density",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/density",
                "unit": "m^-3",
                "llm_cost": None,
            }
        ]
        _call_write(names, mock_gc)
        cypher = _merge_cypher(mock_gc)
        # llm_cost uses an accumulator pattern: adds incoming cost to existing
        # cost when non-null, otherwise preserves existing value.
        assert "sn.llm_cost" in cypher
        assert "coalesce(sn.llm_cost, 0.0) + b.llm_cost" in cypher

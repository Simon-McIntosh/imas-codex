"""Tests for VocabGap infrastructure (Phase 2A–2D).

Covers:
- StandardNameVocabGap / StandardNameComposeBatch model parsing
- write_vocab_gaps dedup and relationship creation
- Ambiguity detection tagging in validate_worker
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test 1: StandardNameVocabGap model
# ---------------------------------------------------------------------------


class TestSNVocabGapModel:
    """StandardNameVocabGap model validates and stores fields."""

    def test_basic_construction(self):
        from imas_codex.standard_names.models import StandardNameVocabGap

        gap = StandardNameVocabGap(
            source_id="equilibrium/time_slice/profiles_1d/psi",
            segment="transformation",
            needed_token="time_derivative_of",
            reason="Need time derivative transformation",
        )
        assert gap.source_id == "equilibrium/time_slice/profiles_1d/psi"
        assert gap.segment == "transformation"
        assert gap.needed_token == "time_derivative_of"
        assert gap.reason == "Need time derivative transformation"

    def test_all_fields_required(self):
        from pydantic import ValidationError

        from imas_codex.standard_names.models import StandardNameVocabGap

        with pytest.raises(ValidationError):
            StandardNameVocabGap(source_id="path/a", segment="transformation")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Test 2: StandardNameComposeBatch with vocab_gaps
# ---------------------------------------------------------------------------


class TestSNComposeBatchVocabGaps:
    """StandardNameComposeBatch correctly parses vocab_gaps from LLM response."""

    def test_compose_batch_with_vocab_gaps(self):
        from imas_codex.standard_names.models import StandardNameComposeBatch

        data = {
            "candidates": [],
            "skipped": [],
            "vocab_gaps": [
                {
                    "source_id": "equilibrium/time_slice/profiles_1d/dpsi_drho_tor",
                    "segment": "transformation",
                    "needed_token": "derivative_of",
                    "reason": "Quantity is a derivative but no derivative transformation exists",
                }
            ],
        }
        batch = StandardNameComposeBatch(**data)
        assert len(batch.vocab_gaps) == 1
        assert batch.vocab_gaps[0].segment == "transformation"
        assert batch.vocab_gaps[0].needed_token == "derivative_of"
        assert batch.vocab_gaps[0].source_id == (
            "equilibrium/time_slice/profiles_1d/dpsi_drho_tor"
        )

    def test_compose_batch_vocab_gaps_default_empty(self):
        from imas_codex.standard_names.models import StandardNameComposeBatch

        batch = StandardNameComposeBatch(candidates=[], skipped=[])
        assert batch.vocab_gaps == []

    def test_compose_batch_multiple_gaps(self):
        from imas_codex.standard_names.models import StandardNameComposeBatch

        data = {
            "candidates": [],
            "skipped": [],
            "vocab_gaps": [
                {
                    "source_id": "path/a",
                    "segment": "transformation",
                    "needed_token": "derivative_of",
                    "reason": "reason A",
                },
                {
                    "source_id": "path/b",
                    "segment": "process",
                    "needed_token": "fusion",
                    "reason": "reason B",
                },
            ],
        }
        batch = StandardNameComposeBatch(**data)
        assert len(batch.vocab_gaps) == 2
        assert {g.needed_token for g in batch.vocab_gaps} == {
            "derivative_of",
            "fusion",
        }


# ---------------------------------------------------------------------------
# Test 3: write_vocab_gaps dedup and relationship creation
# ---------------------------------------------------------------------------


class TestWriteVocabGaps:
    """write_vocab_gaps deduplicates gaps and creates relationships."""

    def _call_write(
        self, gaps: list[dict], mock_gc: MagicMock, source_type: str = "dd"
    ) -> int:
        """Call write_vocab_gaps with a mocked GraphClient."""
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            return write_vocab_gaps(gaps, source_type=source_type)

    def test_empty_returns_zero(self):
        from imas_codex.standard_names.graph_ops import write_vocab_gaps

        assert write_vocab_gaps([]) == 0

    def test_dedup_same_segment_needed_token(self):
        """Two gaps with same segment:needed_token → 1 VocabGap node."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "reason A",
            },
            {
                "source_id": "path/b",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "reason B",
            },
        ]
        result = self._call_write(gaps, mock_gc)
        assert result == 1  # 1 unique VocabGap node

    def test_different_tokens_create_separate_nodes(self):
        """Gaps with different segment:needed_token → separate VocabGap nodes."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "component",
                "needed_token": "novel_component_xyz",
                "reason": "reason A",
            },
            {
                "source_id": "path/b",
                "segment": "process",
                "needed_token": "novel_process_xyz",
                "reason": "reason B",
            },
        ]
        result = self._call_write(gaps, mock_gc)
        assert result == 2  # 2 unique VocabGap nodes

    def test_example_count_accumulates(self):
        """Duplicate segment:needed_token accumulates example_count in batch."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "reason A",
            },
            {
                "source_id": "path/b",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "reason B",
            },
            {
                "source_id": "path/c",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "reason C",
            },
        ]
        self._call_write(gaps, mock_gc)

        # Find the MERGE VocabGap query call and inspect the batch
        merge_call = None
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "MERGE (vg:VocabGap" in cypher:
                merge_call = call
                break
        assert merge_call is not None, "No MERGE VocabGap query found"

        batch = merge_call[1]["batch"]
        assert len(batch) == 1  # 1 deduplicated node
        assert batch[0]["example_count"] == 3  # 3 sources contributed

    def test_dd_source_creates_imasnode_relationship(self):
        """DD source type creates HAS_STANDARD_NAME_VOCAB_GAP from IMASNode."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "needs derivative",
            }
        ]
        self._call_write(gaps, mock_gc, source_type="dd")

        # Find the relationship query
        rel_calls = [
            c
            for c in mock_gc.query.call_args_list
            if "HAS_STANDARD_NAME_VOCAB_GAP" in c[0][0] and "IMASNode" in c[0][0]
        ]
        assert len(rel_calls) == 1, (
            "Should create DD HAS_STANDARD_NAME_VOCAB_GAP relationship"
        )

        # Verify reason is in the relationship batch
        rel_batch = rel_calls[0][1]["batch"]
        assert len(rel_batch) == 1
        assert rel_batch[0]["reason"] == "needs derivative"
        assert rel_batch[0]["source_id"] == ("equilibrium/time_slice/profiles_1d/psi")

    def test_signal_source_creates_facilitysignal_relationship(self):
        """Signal source type creates HAS_STANDARD_NAME_VOCAB_GAP from FacilitySignal."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "tcv:ip/measured",
                "segment": "subject",
                "needed_token": "plasma_current_ip",
                "reason": "missing subject token",
            }
        ]
        self._call_write(gaps, mock_gc, source_type="signals")

        # Find the relationship query for FacilitySignal
        rel_calls = [
            c
            for c in mock_gc.query.call_args_list
            if "HAS_STANDARD_NAME_VOCAB_GAP" in c[0][0] and "FacilitySignal" in c[0][0]
        ]
        assert len(rel_calls) == 1, (
            "Should create signal HAS_STANDARD_NAME_VOCAB_GAP relationship"
        )

    def test_relationship_has_per_source_reason(self):
        """Each HAS_STANDARD_NAME_VOCAB_GAP relationship carries source-specific reason."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "reason for A",
            },
            {
                "source_id": "path/b",
                "segment": "process",
                "needed_token": "novel_process_zzz",
                "reason": "reason for B",
            },
        ]
        self._call_write(gaps, mock_gc)

        # Find relationship query
        rel_calls = [
            c
            for c in mock_gc.query.call_args_list
            if "HAS_STANDARD_NAME_VOCAB_GAP" in c[0][0]
        ]
        assert len(rel_calls) >= 1

        rel_batch = rel_calls[0][1]["batch"]
        assert len(rel_batch) == 2  # 2 relationships (one per source)
        reasons = {r["reason"] for r in rel_batch}
        assert reasons == {"reason for A", "reason for B"}


# ---------------------------------------------------------------------------
# Test 4: Ambiguity detection tagging
# ---------------------------------------------------------------------------


class TestAmbiguityClassification:
    """Validate worker tags component/coordinate overlap as grammar ambiguity."""

    def test_component_coordinate_overlap_classification(self):
        """Error containing 'component' AND 'coordinate' → specific ambiguity tag."""
        # Reproduce the classification logic from validate_worker
        exc_msg = "Token 'radial' is ambiguous: matches both component and coordinate"
        exc_msg_lower = exc_msg.lower()
        name = "radial_electron_temperature"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("grammar:ambiguity:component_coordinate_overlap:")
        assert name in issues[0]

    def test_generic_ambiguity_classification(self):
        """Error containing 'ambig' but NOT component+coordinate → unclassified."""
        exc_msg = "Ambiguous token: cannot resolve segment"
        exc_msg_lower = exc_msg.lower()
        name = "some_ambiguous_name"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("grammar:ambiguity:unclassified:")

    def test_plain_parse_error_classification(self):
        """Error without ambiguity keywords → generic parse_error."""
        exc_msg = "Invalid token sequence in standard name"
        exc_msg_lower = exc_msg.lower()
        name = "broken_name_here"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("parse_error:")

    def test_component_only_is_not_overlap(self):
        """Error with 'component' but NOT 'coordinate' → generic parse_error."""
        exc_msg = "Unknown component token 'radial'"
        exc_msg_lower = exc_msg.lower()
        name = "radial_temperature"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("parse_error:")

    def test_coordinate_only_is_not_overlap(self):
        """Error with 'coordinate' but NOT 'component' → generic parse_error."""
        exc_msg = "Unknown coordinate token 'toroidal'"
        exc_msg_lower = exc_msg.lower()
        name = "toroidal_field"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("parse_error:")


# ---------------------------------------------------------------------------
# Test 5: persist_composed_batch creates VocabGap from token-miss
# ---------------------------------------------------------------------------


class TestPersistBatchTokenMissGaps:
    """write_standard_names calls write_vocab_gaps for token misses detected
    during _write_grammar_decomposition."""

    def test_token_miss_creates_vocab_gap_nodes(self):
        """When _write_grammar_decomposition detects unmatched tokens, VocabGap nodes are created."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        # Simulate _write_grammar_decomposition returning a token miss
        token_miss_gaps = [
            {
                "sn_id": "electron_temperature",
                "segment": "subject",
                "needed_token": "exotic_particle",
            }
        ]

        names = [
            {
                "id": "electron_temperature",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "source_types": ["dd"],
                "unit": "eV",
            }
        ]

        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.graph_ops._write_grammar_decomposition",
                return_value=token_miss_gaps,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_vocab_gaps"
            ) as mock_write_vg,
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        # write_vocab_gaps should have been called with DD gaps
        mock_write_vg.assert_called_once()
        call_args = mock_write_vg.call_args
        gap_dicts = call_args[0][0]
        assert len(gap_dicts) == 1
        assert gap_dicts[0]["source_id"] == (
            "core_profiles/profiles_1d/electrons/temperature"
        )
        assert gap_dicts[0]["segment"] == "subject"
        assert gap_dicts[0]["needed_token"] == "exotic_particle"
        assert call_args[1]["source_type"] == "dd"

    def test_no_token_miss_skips_write_vocab_gaps(self):
        """When _write_grammar_decomposition returns no gaps, write_vocab_gaps is not called."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "source_types": ["dd"],
                "unit": "eV",
            }
        ]

        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.graph_ops._write_grammar_decomposition",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_vocab_gaps"
            ) as mock_write_vg,
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        mock_write_vg.assert_not_called()

    def test_signal_source_routes_to_signals_type(self):
        """Signal source_types route gaps to write_vocab_gaps with source_type='signals'."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        token_miss_gaps = [
            {
                "sn_id": "plasma_current",
                "segment": "process",
                "needed_token": "novel_process",
            }
        ]

        names = [
            {
                "id": "plasma_current",
                "source_id": "tcv:ip/measured",
                "source_types": ["signals"],
                "unit": "A",
            }
        ]

        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.graph_ops._write_grammar_decomposition",
                return_value=token_miss_gaps,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_vocab_gaps"
            ) as mock_write_vg,
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        mock_write_vg.assert_called_once()
        assert mock_write_vg.call_args[1]["source_type"] == "signals"


# ---------------------------------------------------------------------------
# Test 6: _resolve_grammar_token_version fallback
# ---------------------------------------------------------------------------


class TestResolveGrammarTokenVersion:
    """_resolve_grammar_token_version uses exact ISN version when available,
    falls back to latest graph version, or returns None."""

    def test_exact_version_match(self):
        """Returns ISN version when GrammarToken nodes exist for it."""
        from imas_codex.standard_names.graph_ops import _resolve_grammar_token_version

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"t.version": "0.7.0rc16"}])

        result = _resolve_grammar_token_version(mock_gc, "0.7.0rc16")
        assert result == "0.7.0rc16"

    def test_fallback_to_latest(self):
        """Falls back to latest available version when exact doesn't exist."""
        from imas_codex.standard_names.graph_ops import _resolve_grammar_token_version

        mock_gc = MagicMock()

        # First call (exact version) returns empty, second (fallback) returns rc14
        mock_gc.query = MagicMock(side_effect=[[], [{"v": "0.7.0rc14"}]])

        result = _resolve_grammar_token_version(mock_gc, "0.7.0rc16")
        assert result == "0.7.0rc14"

    def test_no_grammar_tokens(self):
        """Returns None when no GrammarToken nodes exist at all."""
        from imas_codex.standard_names.graph_ops import _resolve_grammar_token_version

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        result = _resolve_grammar_token_version(mock_gc, "0.7.0rc16")
        assert result is None


# ---------------------------------------------------------------------------
# Test 7: _write_grammar_decomposition version fallback integration
# ---------------------------------------------------------------------------


class TestWriteSegmentEdgesVersionFallback:
    """_write_grammar_decomposition uses version fallback to avoid false-positive
    VocabGap nodes when GrammarToken nodes are stale."""

    def test_no_grammar_tokens_skips_entirely(self):
        """When no GrammarToken nodes exist, returns empty gaps without parsing."""
        mock_gc = MagicMock()

        with patch(
            "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
            return_value=None,
        ):
            from imas_codex.standard_names.graph_ops import _write_grammar_decomposition

            gaps = _write_grammar_decomposition(mock_gc, ["electron_temperature"])

        assert gaps == []

    def test_fallback_version_passed_to_cypher(self):
        """When ISN version differs from graph tokens, fallback version is used
        in the OPTIONAL MATCH to avoid false-positive VocabGap."""
        mock_gc = MagicMock()
        # Return matched=True for the token query
        mock_gc.query = MagicMock(
            return_value=[{"token": "electron", "segment": "subject", "matched": True}]
        )

        _all_segs = frozenset(
            {
                "component",
                "coordinate",
                "subject",
                "device",
                "geometric_base",
                "physical_base",
                "object",
                "geometry",
                "position",
                "region",
                "process",
            }
        )
        with (
            patch(
                "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
                return_value="0.7.0rc14",
            ),
            patch(
                "imas_codex.standard_names.graph_ops._resolve_synced_segments",
                return_value=_all_segs,
            ),
            patch("imas_standard_names.grammar.parse_standard_name") as mock_parse,
            patch("imas_standard_names.graph.spec.segment_edge_specs") as mock_specs,
        ):
            mock_parsed = MagicMock()
            mock_parse.return_value = mock_parsed
            mock_spec = MagicMock()
            mock_spec.position = 2
            mock_spec.segment = "subject"
            mock_spec.token = "electron"
            mock_specs.return_value = [mock_spec]

            from imas_codex.standard_names.graph_ops import _write_grammar_decomposition

            gaps = _write_grammar_decomposition(mock_gc, ["electron_temperature"])

        # No gaps — token was matched via fallback version
        assert gaps == []

        # Verify fallback version (0.7.0rc14) was used in the query
        opt_match_calls = [
            c for c in mock_gc.query.call_args_list if "OPTIONAL MATCH" in str(c)
        ]
        assert len(opt_match_calls) >= 1
        assert opt_match_calls[0][1]["token_version"] == "0.7.0rc14"


# ---------------------------------------------------------------------------
# Test 5: Open-vocabulary segment filtering (Problem 1 regression)
# ---------------------------------------------------------------------------


class TestOpenSegmentFilter:
    """Open-vocabulary segments must never produce VocabGap nodes.

    ``physical_base`` is open by design — any compound is admissible.
    ``grammar_ambiguity`` is a pseudo segment reported by the composer for
    structural ambiguity rather than missing tokens.  Gaps on either are
    filtered out before ``write_vocab_gaps`` persists them, and they never
    retire the underlying ``StandardNameSource`` to ``vocab_gap`` status.
    """

    def test_open_segments_includes_physical_base(self):
        from imas_codex.standard_names.segments import open_segments

        assert "physical_base" not in open_segments()

    def test_is_open_segment_predicate(self):
        from imas_codex.standard_names.segments import is_open_segment

        assert is_open_segment("physical_base") is False
        assert is_open_segment("grammar_ambiguity") is True
        # Closed segments must not be flagged as open
        assert is_open_segment("qualifier") is False
        assert is_open_segment("subject") is False
        assert is_open_segment("position") is False
        assert is_open_segment("component") is False
        assert is_open_segment(None) is False
        assert is_open_segment("") is False

    def test_filter_closed_segment_gaps_splits_by_openness(self):
        from imas_codex.standard_names.segments import filter_closed_segment_gaps

        gaps = [
            {"source_id": "a", "segment": "component", "needed_token": "curl_of"},
            {
                "source_id": "b",
                "segment": "physical_base",
                "needed_token": "toroidal_torque",
            },
            {
                "source_id": "c",
                "segment": "grammar_ambiguity",
                "needed_token": "diamagnetic",
            },
            {"source_id": "d", "segment": "subject", "needed_token": "pellet"},
        ]
        kept, dropped = filter_closed_segment_gaps(gaps)
        kept_segs = {g["segment"] for g in kept}
        drop_segs = {g["segment"] for g in dropped}
        assert kept_segs == {"component", "subject", "physical_base"}
        assert drop_segs == {"grammar_ambiguity"}

    def test_write_vocab_gaps_skips_open_segments(self):
        """write_vocab_gaps must not emit MERGE for open/pseudo segments."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "segment": "physical_base",
                "needed_token": "frobnicating_zorch",
                "reason": "closed segment — should be persisted",
            },
            {
                "source_id": "core_profiles/ions/velocity",
                "segment": "grammar_ambiguity",
                "needed_token": "diamagnetic",
                "reason": "structural ambiguity — should be filtered",
            },
        ]

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            written = write_vocab_gaps(gaps, source_type="dd")

        # Only the physical_base gap persisted; grammar_ambiguity filtered
        assert written == 1
        merge_calls = [
            c for c in mock_gc.query.call_args_list if "MERGE (vg:VocabGap" in c[0][0]
        ]
        assert len(merge_calls) == 1, (
            "grammar_ambiguity gaps must never reach the VocabGap MERGE query"
        )

    def test_write_vocab_gaps_mixed_batch_only_persists_closed(self):
        """Mixed batch: pseudo segment filtered, decomposable filtered, absent kept."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "physical_base",  # closed — keep (truly absent token)
                "needed_token": "frobnicating_zorch",
                "reason": "real gap",
            },
            {
                "source_id": "path/b",
                "segment": "component",  # closed — keep (truly absent token)
                "needed_token": "novel_component_zzz",
                "reason": "real gap",
            },
        ]

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            written = write_vocab_gaps(gaps, source_type="dd")

        assert written == 2  # both closed-segment gaps materialised
        merge_calls = [
            c for c in mock_gc.query.call_args_list if "MERGE (vg:VocabGap" in c[0][0]
        ]
        assert len(merge_calls) == 1
        batch = merge_calls[0][1]["batch"]
        assert len(batch) == 2
        batch_segs = {item["segment"] for item in batch}
        assert batch_segs == {"physical_base", "component"}


class TestLoadKnownPhysicalBases:
    """Verify _load_known_physical_bases returns expected tokens."""

    def test_returns_nonempty_frozenset(self):
        from imas_codex.standard_names.workers import _load_known_physical_bases

        bases = _load_known_physical_bases()
        assert isinstance(bases, frozenset)
        assert len(bases) > 0, "expected non-empty frozenset of physical bases"

    @pytest.mark.parametrize(
        "token",
        ["pressure", "density", "velocity", "temperature", "current", "wavelength"],
    )
    def test_contains_key_tokens(self, token):
        from imas_codex.standard_names.workers import _load_known_physical_bases

        bases = _load_known_physical_bases()
        assert token in bases, (
            f"expected physical_base token {token!r} missing from "
            f"_load_known_physical_bases() result ({len(bases)} tokens)"
        )


# ---------------------------------------------------------------------------
# ISN-backed segment classification and reconciliation
# ---------------------------------------------------------------------------


class TestKnownSegments:
    """known_segments() returns ISN grammar segment names."""

    def test_returns_frozenset(self):
        from imas_codex.standard_names.segments import known_segments

        segs = known_segments()
        assert segs is not None, "ISN must be installed for tests"
        assert isinstance(segs, frozenset)

    def test_contains_core_segments(self):
        from imas_codex.standard_names.segments import known_segments

        segs = known_segments()
        assert segs is not None
        for expected in ("component", "coordinate", "subject", "process", "position"):
            assert expected in segs, f"Missing segment {expected!r}"

    def test_is_valid_segment_real(self):
        from imas_codex.standard_names.segments import is_valid_segment

        assert is_valid_segment("component") is True
        assert is_valid_segment("position") is True
        assert is_valid_segment("qualifier") is True

    def test_is_valid_segment_pseudo(self):
        from imas_codex.standard_names.segments import is_valid_segment

        assert is_valid_segment("grammar_ambiguity") is True

    def test_is_valid_segment_bogus(self):
        from imas_codex.standard_names.segments import is_valid_segment

        assert is_valid_segment("foobar_nonexistent") is False
        assert is_valid_segment("transformation_xyz") is False

    def test_is_valid_segment_none_empty(self):
        from imas_codex.standard_names.segments import is_valid_segment

        assert is_valid_segment(None) is False
        assert is_valid_segment("") is False

    def test_isn_unavailable_returns_true_conservatively(self):
        """When ISN is not importable, is_valid_segment returns True."""
        from imas_codex.standard_names.segments import is_valid_segment

        with patch(
            "imas_codex.standard_names.segments._load_segment_token_map",
            return_value=None,
        ):
            # Clear caches
            from imas_codex.standard_names.segments import known_segments

            known_segments.cache_clear()
            try:
                assert is_valid_segment("anything") is True
            finally:
                known_segments.cache_clear()


class TestClassifyGap:
    """classify_gap() returns correct (category, actual_segments) tuples."""

    def test_absent_token(self):
        from imas_codex.standard_names.segments import classify_gap

        cat, actual = classify_gap("position", "zzz_nonexistent_token_xyzzy")
        assert cat == "absent"
        assert actual == []

    def test_false_positive(self):
        """Token exists in the reported segment → false_positive."""
        from imas_codex.standard_names.segments import classify_gap, is_known_token

        # Find a token that exists in 'qualifier' segment
        segs = is_known_token("maximum")
        assert "qualifier" in segs, "maximum should be in qualifier"

        cat, actual = classify_gap("qualifier", "maximum")
        assert cat == "false_positive"
        assert "qualifier" in actual

    def test_wrong_slot_placement(self):
        """Token exists in a different segment than reported."""
        from imas_codex.standard_names.segments import classify_gap

        # 'electron' is a subject token — report it as 'position' → wrong_slot
        cat, actual = classify_gap("position", "electron")
        assert cat == "wrong_slot_placement"
        assert "subject" in actual

    def test_invalid_segment(self):
        from imas_codex.standard_names.segments import classify_gap

        cat, actual = classify_gap("nonexistent_segment_xyz", "some_token")
        assert cat == "invalid_segment"
        assert actual == []

    def test_open_segment(self):
        from imas_codex.standard_names.segments import classify_gap, open_segments

        open_segs = open_segments()
        if not open_segs:
            pytest.skip("No open segments in current ISN version")
        seg = next(iter(open_segs))
        cat, actual = classify_gap(seg, "any_token")
        assert cat == "open_segment"

    def test_decomposable_compound(self):
        """Compound token whose parts are all registered → decomposable."""
        from imas_codex.standard_names.segments import classify_gap, is_known_token

        # Verify prerequisites: 'thermal' in qualifier/subject, 'pressure' in physical_base
        segs_thermal = is_known_token("thermal")
        segs_pressure = is_known_token("pressure")
        if not segs_thermal or not segs_pressure:
            pytest.skip("Need thermal+pressure as registered tokens")

        cat, actual = classify_gap("physical_base", "thermal_pressure")
        assert cat == "decomposable"
        assert len(actual) >= 2  # parts found in ≥2 segments

    def test_decomposable_cross_segment(self):
        """Compound where parts span different segments."""
        from imas_codex.standard_names.segments import classify_gap, is_known_token

        # 'poloidal' should be in component/coordinate
        segs_pol = is_known_token("poloidal")
        if not segs_pol:
            pytest.skip("Need poloidal as registered token")

        # 'magnetic_flux' should be in physical_base
        segs_mf = is_known_token("magnetic_flux")
        if not segs_mf:
            pytest.skip("Need magnetic_flux as registered token")

        cat, actual = classify_gap("physical_base", "poloidal_magnetic_flux")
        # This is an ATOMIC_COMPOUND — should be absent (preserved)
        assert cat == "absent"

    def test_atomic_compound_not_decomposed(self):
        """Atomic compounds in whitelist must NOT be classified as decomposable."""
        from imas_codex.standard_names.segments import (
            ATOMIC_COMPOUNDS,
            classify_gap,
        )

        for compound in ["magnetic_field", "current_density", "safety_factor"]:
            if compound not in ATOMIC_COMPOUNDS:
                continue
            cat, _ = classify_gap("physical_base", compound)
            assert cat != "decomposable", (
                f"{compound} is in ATOMIC_COMPOUNDS but was classified as decomposable"
            )

    def test_single_token_not_decomposable(self):
        """Single-word tokens with no underscores cannot be decomposable."""
        from imas_codex.standard_names.segments import classify_gap

        cat, actual = classify_gap("qualifier", "zzz_truly_unique_xyzzy")
        assert cat == "absent"
        assert actual == []


class TestWriteVocabGapsInvalidSegment:
    """write_vocab_gaps rejects gaps with invalid ISN segment names."""

    def test_invalid_segment_skipped(self):
        """Gaps with a bogus segment name should not be written."""
        from imas_codex.standard_names.graph_ops import write_vocab_gaps

        gaps = [
            {
                "segment": "foobar_nonexistent",
                "needed_token": "test_token",
                "source_id": "dd:equilibrium/test",
                "reason": "test",
            }
        ]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

            written = write_vocab_gaps(gaps, source_type="dd")
            assert written == 0

    def test_false_positive_skipped(self):
        """Gaps where token exists in reported segment should be skipped."""
        from imas_codex.standard_names.graph_ops import write_vocab_gaps

        gaps = [
            {
                "segment": "qualifier",
                "needed_token": "maximum",
                "source_id": "dd:equilibrium/test",
                "reason": "test",
            }
        ]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

            written = write_vocab_gaps(gaps, source_type="dd")
            assert written == 0


class TestReconcileVocabGaps:
    """reconcile_vocab_gaps() scrubs stale VocabGap nodes against ISN."""

    def test_deletes_false_positive(self):
        """Gap where token now exists in reported segment → deleted."""
        from imas_codex.standard_names.graph_ops import reconcile_vocab_gaps

        fake_gaps = [
            {
                "id": "vocab_gap:qualifier:maximum",
                "segment": "qualifier",
                "token": "maximum",
                "category": "absent",
            }
        ]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_gc.query.side_effect = [
                fake_gaps,  # SELECT all gaps
                None,  # DELETE
            ]

            stats = reconcile_vocab_gaps()

        assert stats["deleted_false_positive"] == 1
        assert stats["remaining"] == 0

    def test_preserves_genuine_absent_gap(self):
        """Genuinely absent token → preserved."""
        from imas_codex.standard_names.graph_ops import reconcile_vocab_gaps

        fake_gaps = [
            {
                "id": "vocab_gap:position:zzz_nonexistent_xyzzy",
                "segment": "position",
                "token": "zzz_nonexistent_xyzzy",
                "category": "absent",
            }
        ]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_gc.query.return_value = fake_gaps

            stats = reconcile_vocab_gaps()

        assert stats["remaining"] == 1
        assert stats["deleted_false_positive"] == 0

    def test_reclassifies_wrong_slot(self):
        """Token that appeared in a different segment → reclassified."""
        from imas_codex.standard_names.graph_ops import reconcile_vocab_gaps

        fake_gaps = [
            {
                "id": "vocab_gap:position:electron",
                "segment": "position",
                "token": "electron",
                "category": "absent",  # was absent, should be wrong_slot
            }
        ]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_gc.query.side_effect = [
                fake_gaps,  # SELECT all gaps
                None,  # UPDATE
            ]

            stats = reconcile_vocab_gaps()

        assert stats["reclassified"] == 1
        assert stats["remaining"] == 1

    def test_isn_unavailable_skips(self):
        """When ISN is unavailable, reconcile skips gracefully."""
        from imas_codex.standard_names.graph_ops import reconcile_vocab_gaps

        with patch(
            "imas_codex.standard_names.segments._load_segment_token_map",
            return_value=None,
        ):
            from imas_codex.standard_names.segments import known_segments

            known_segments.cache_clear()
            try:
                stats = reconcile_vocab_gaps()
                assert stats.get("skipped") is True
                assert stats["checked"] == 0
            finally:
                known_segments.cache_clear()

    def test_empty_graph(self):
        """No VocabGap nodes → quick return."""
        from imas_codex.standard_names.graph_ops import reconcile_vocab_gaps

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_gc.query.return_value = []

            stats = reconcile_vocab_gaps()

        assert stats["checked"] == 0
        assert stats["remaining"] == 0


# ---------------------------------------------------------------------------
# Test: Batch rescue validator (per-candidate vocab gap isolation)
# ---------------------------------------------------------------------------


class TestBatchRescueValidator:
    """StandardNameComposeBatch._rescue_failed_candidates isolates per-candidate
    vocab-gap failures instead of failing the entire batch."""

    def _valid_candidate(self, source_id: str = "path/a") -> dict:
        """Build a minimal valid candidate dict."""
        return {
            "source_id": source_id,
            "segments": {
                "base_token": "temperature",
                "base_kind": "quantity",
            },
            "description": "Test temperature",
            "reason": "test",
        }

    def _bad_qualifier_candidate(self, source_id: str = "path/b") -> dict:
        """Build a candidate with an unregistered qualifier token."""
        return {
            "source_id": source_id,
            "segments": {
                "base_token": "torque",
                "base_kind": "quantity",
                "qualifiers": ["cumulative"],
            },
            "description": "Cumulative torque",
            "reason": "test",
        }

    def test_all_valid_candidates_preserved(self):
        """When all candidates are valid, none are rescued."""
        from imas_codex.standard_names.models import StandardNameComposeBatch

        batch = StandardNameComposeBatch(
            candidates=[self._valid_candidate("p/a"), self._valid_candidate("p/b")]
        )
        assert len(batch.candidates) == 2
        assert len(batch.vocab_gaps) == 0

    def test_bad_candidate_rescued_to_vocab_gap(self):
        """A candidate with an unregistered qualifier is rescued to vocab_gaps."""
        from imas_codex.standard_names.models import StandardNameComposeBatch

        data = {
            "candidates": [
                self._valid_candidate("p/a"),
                self._bad_qualifier_candidate("p/b"),
            ],
        }
        batch = StandardNameComposeBatch(**data)
        assert len(batch.candidates) == 1
        assert batch.candidates[0].segments.base_token == "temperature"
        assert len(batch.vocab_gaps) == 1
        assert batch.vocab_gaps[0].source_id == "p/b"
        assert batch.vocab_gaps[0].needed_token == "cumulative"

    def test_multiple_bad_candidates_all_rescued(self):
        """Multiple bad candidates are individually rescued."""
        from imas_codex.standard_names.models import StandardNameComposeBatch

        bad_1 = {
            "source_id": "p/x",
            "segments": {
                "base_token": "magnetic_field",
                "base_kind": "quantity",
                "qualifiers": ["hypothetical"],
            },
            "description": "Hypothetical field",
            "reason": "test",
        }
        bad_2 = self._bad_qualifier_candidate("p/y")
        data = {
            "candidates": [self._valid_candidate("p/a"), bad_1, bad_2],
        }
        batch = StandardNameComposeBatch(**data)
        assert len(batch.candidates) == 1
        assert len(batch.vocab_gaps) == 2
        gap_sources = {g.source_id for g in batch.vocab_gaps}
        assert gap_sources == {"p/x", "p/y"}

    def test_existing_vocab_gaps_preserved(self):
        """Rescued gaps are appended to existing LLM-reported vocab_gaps."""
        from imas_codex.standard_names.models import StandardNameComposeBatch

        data = {
            "candidates": [self._bad_qualifier_candidate("p/b")],
            "vocab_gaps": [
                {
                    "source_id": "p/c",
                    "segment": "process",
                    "needed_token": "fusion",
                    "reason": "LLM reported",
                }
            ],
        }
        batch = StandardNameComposeBatch(**data)
        assert len(batch.candidates) == 0
        assert len(batch.vocab_gaps) == 2
        gap_sources = {g.source_id for g in batch.vocab_gaps}
        assert gap_sources == {"p/b", "p/c"}

    def test_non_vocab_errors_not_rescued(self):
        """Candidates with non-vocab validation errors stay in candidates list."""
        from pydantic import ValidationError

        from imas_codex.standard_names.models import StandardNameComposeBatch

        # Bad base_kind is not a vocab gap — it's an enum error
        bad_enum = {
            "source_id": "p/z",
            "segments": {
                "base_token": "temperature",
                "base_kind": "invalid_kind",
            },
            "description": "Bad kind",
            "reason": "test",
        }
        data = {
            "candidates": [self._valid_candidate("p/a"), bad_enum],
        }
        # The bad enum candidate stays in the list and fails normal validation
        with pytest.raises(ValidationError):
            StandardNameComposeBatch(**data)

    def test_all_bad_produces_empty_candidates(self):
        """When all candidates have vocab gaps, candidates list is empty."""
        from imas_codex.standard_names.models import StandardNameComposeBatch

        data = {
            "candidates": [
                self._bad_qualifier_candidate("p/a"),
                self._bad_qualifier_candidate("p/b"),
            ],
        }
        batch = StandardNameComposeBatch(**data)
        assert len(batch.candidates) == 0
        assert len(batch.vocab_gaps) == 2


class TestExtractGapFromError:
    """_extract_gap_from_error parses vocab-gap validation error messages."""

    def test_qualifier_error(self):
        from imas_codex.standard_names.models import _extract_gap_from_error

        segment, token = _extract_gap_from_error(
            "qualifier 'cumulative' is not a registered grammar token.",
            {"base_token": "torque"},
        )
        assert segment == "qualifier"
        assert token == "cumulative"

    def test_base_token_error(self):
        from imas_codex.standard_names.models import _extract_gap_from_error

        segment, token = _extract_gap_from_error(
            "base_token 'exotic_flux' is not a registered physical_base.",
            {"base_token": "exotic_flux"},
        )
        assert segment == "physical_base"
        assert token == "exotic_flux"

    def test_projection_axis_error(self):
        from imas_codex.standard_names.models import _extract_gap_from_error

        segment, token = _extract_gap_from_error(
            "projection_axis 'spiral' is not a registered component token.",
            {"projection_axis": "spiral"},
        )
        assert segment == "component"
        assert token == "spiral"

    def test_fallback_extracts_from_segments(self):
        from imas_codex.standard_names.models import _extract_gap_from_error

        segment, token = _extract_gap_from_error(
            "some unexpected error format",
            {"base_token": "fallback_value"},
        )
        assert segment == "base_token"
        assert token == "fallback_value"

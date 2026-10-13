"""Tests for the declared domain order and graph-writer compatibility helpers."""

import pytest

from imas_codex.standard_names import domain_ranking as dr


class TestDeclaredOrder:
    def test_more_central_has_lower_rank(self):
        rank = dr.FALLBACK_RANK_TABLE.index
        assert rank("equilibrium") < rank("transport") < rank("general")

    def test_general_is_last(self):
        assert dr.FALLBACK_RANK_TABLE[-1] == "general"


class TestPromotionAdapter:
    @pytest.mark.parametrize(
        "current,candidate,expected",
        [
            (None, None, None),
            (None, "equilibrium", "equilibrium"),
            ("equilibrium", None, "equilibrium"),
            ("", "equilibrium", "equilibrium"),
            ("equilibrium", "", "equilibrium"),
            ("transport", "equilibrium", "equilibrium"),
            ("general", "magnetohydrodynamics", "magnetohydrodynamics"),
            ("plasma_control", "equilibrium", "equilibrium"),
            ("equilibrium", "transport", "equilibrium"),
            ("magnetohydrodynamics", "general", "magnetohydrodynamics"),
            ("equilibrium", "equilibrium", "equilibrium"),
        ],
    )
    def test_uses_declared_selector(self, current, candidate, expected):
        assert dr.maybe_promote_domain(current, candidate) == expected

    def test_refuses_unknown(self):
        with pytest.raises(ValueError, match="unknown_domain"):
            dr.maybe_promote_domain("unknown_domain", "equilibrium")


class TestMergeSourceDomains:
    def test_empty_inputs(self):
        assert dr.merge_source_domains(None) == []
        assert dr.merge_source_domains([]) == []
        assert dr.merge_source_domains(None, None) == []

    def test_appends_domain(self):
        assert dr.merge_source_domains(["equilibrium"], "transport") == [
            "equilibrium",
            "transport",
        ]

    def test_dedupes(self):
        assert dr.merge_source_domains(["equilibrium"], "equilibrium", "transport") == [
            "equilibrium",
            "transport",
        ]

    def test_preserves_existing_order(self):
        assert dr.merge_source_domains(["transport", "equilibrium"], "general") == [
            "transport",
            "equilibrium",
            "general",
        ]

    def test_skips_empty_and_none(self):
        assert dr.merge_source_domains(["equilibrium"], None, "", "transport") == [
            "equilibrium",
            "transport",
        ]

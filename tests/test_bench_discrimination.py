"""Reviewer-discrimination bench metrics — pure-function unit tests.

Lives at tests/ root (not tests/standard_names/) so it does not trigger that
package's autouse grammar-context fixture, which loads the full ISN catalog.
These tests exercise only pure scoring/seed helpers and need no catalog.
"""

from __future__ import annotations

from imas_codex.standard_names import benchmark_roles as br


class TestDiscriminationMetrics:
    def test_perfect_separation(self):
        m = br.discrimination_metrics([0.9, 0.85, 0.8], [0.3, 0.4, 0.5])
        assert m["separation"] > 0
        assert m["auc"] == 1.0
        assert m["bad_recall"] == 1.0
        assert m["good_pass"] == 1.0

    def test_no_discrimination(self):
        # Identical distributions → AUC 0.5, zero separation.
        m = br.discrimination_metrics([0.6, 0.6], [0.6, 0.6])
        assert m["separation"] == 0.0
        assert m["auc"] == 0.5

    def test_inverted_reviewer_scores_below_half_auc(self):
        # A reviewer that scores bad items HIGHER than good is worse than chance.
        m = br.discrimination_metrics([0.3, 0.4], [0.8, 0.9])
        assert m["auc"] < 0.5
        assert m["separation"] < 0

    def test_recall_and_pass_at_threshold(self):
        # threshold 0.75: two bad below (caught), one bad above (missed);
        # two good above (kept), one good below (falsely rejected).
        m = br.discrimination_metrics([0.9, 0.8, 0.6], [0.5, 0.7, 0.85], threshold=0.75)
        assert m["bad_recall"] == round(2 / 3, 4)
        assert m["good_pass"] == round(2 / 3, 4)

    def test_empty_inputs(self):
        assert br.discrimination_metrics([], []) == {}


class TestSeedDefects:
    def test_banned_prose_injects_padding(self):
        out = br._seed_bad_documentation("Base definition.", "banned_prose", "W.m^-2")
        assert "Base definition." in out and "Typical values" in out

    def test_vacuous_replaces_with_tautology(self):
        out = br._seed_bad_documentation("Real physics.", "vacuous", "W")
        assert "Real physics." not in out

    def test_unit_contradiction_cites_registered_unit(self):
        out = br._seed_bad_documentation("Def.", "unit_contradiction", "W.m^-2")
        assert "W.m^-2" in out and "dimensionless" in out

    def test_defect_types_cover_the_seed_cycle(self):
        assert set(br._DISCRIM_DEFECTS) == {
            "banned_prose",
            "vacuous",
            "unit_contradiction",
        }

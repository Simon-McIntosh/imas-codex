"""Known-answer tests for the shared evaluation module.

Each metric is pinned against an answer known independently of it: a rank AUC
that fully separates the classes is 1.0, its reverse 0.0, a fully tied ranking
0.5, and a percentile interval brackets the estimate it was built from with
real width.  The operating point is checked to refuse the split it chose on.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.discovery.base import evaluation as ev


def _fixture_row(**overrides):
    row = {
        "unit": "u0",
        "label": 1,
        "canonical_unit": "u0",
        "copy_group": "g0",
        "label_source": "src",
        "source_evidence": "ev",
        "split": "validation",
    }
    row.update(overrides)
    return row


def _separable_arm():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.7, 0.6, 0.2, 0.8, 0.3, 0.15, 0.1]
    return labels, scores


class TestRocAuc:
    def test_perfect_separation_scores_one(self):
        assert ev.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0

    def test_reversed_scores_score_zero(self):
        assert ev.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0

    def test_all_tied_scores_score_one_half(self):
        assert ev.roc_auc([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5

    def test_partial_overlap_is_strictly_inside(self):
        labels, scores = _separable_arm()
        assert 0.0 < ev.roc_auc(labels, scores) < 1.0


class TestBootstrapInterval:
    def test_interval_contains_its_point_estimate(self):
        labels, scores = _separable_arm()
        out = ev.bootstrap_interval(labels, scores, seed=7)
        assert out["point"] == ev.roc_auc(labels, scores)
        assert out["lower"] < out["upper"]
        assert out["lower"] <= out["point"] <= out["upper"]
        assert out["seed"] == 7
        assert out["n_resamples"] == 1000


class TestPairedAuc:
    def test_difference_is_arm_a_minus_arm_b(self):
        labels = [1, 0, 1, 0]
        a = [0.9, 0.1, 0.8, 0.2]
        b = [0.9, 0.1, 0.2, 0.8]
        out = ev.paired_auc(labels, a, b)
        assert out["auc_a"] == 1.0
        assert out["difference"] == out["auc_a"] - out["auc_b"]
        assert out["difference"] > 0


class TestOperatingPoint:
    def _rows(self):
        return [
            _fixture_row(
                unit="c1", split="calibration", label=1, score=0.9, copy_group="c1"
            ),
            _fixture_row(
                unit="c2", split="calibration", label=0, score=0.2, copy_group="c2"
            ),
            _fixture_row(
                unit="v1", split="validation", label=1, score=0.95, copy_group="v1"
            ),
            _fixture_row(
                unit="v2", split="validation", label=0, score=0.1, copy_group="v2"
            ),
        ]

    def test_refuses_to_report_on_its_calibration_split(self):
        with pytest.raises(ValueError, match="calibration"):
            ev.choose_operating_point(
                self._rows(),
                cost_false_positive=1.0,
                cost_false_negative=1.0,
                report_split="calibration",
            )

    def test_chooses_on_calibration_reports_on_validation(self):
        out = ev.choose_operating_point(
            self._rows(), cost_false_positive=1.0, cost_false_negative=1.0
        )
        assert out["split"] == "validation"
        assert out["report"]["accuracy"] == 1.0


class TestFixtureLoader:
    def test_loads_rows_that_carry_every_field(self, tmp_path):
        path = tmp_path / "fixture.json"
        path.write_text(json.dumps([_fixture_row(), _fixture_row(unit="u1")]))
        assert len(ev.load_labelled_fixture(path))

    def test_refuses_a_row_missing_a_field_naming_it_and_its_index(self, tmp_path):
        row = _fixture_row()
        del row["canonical_unit"]
        path = tmp_path / "fixture.json"
        path.write_text(json.dumps([_fixture_row(), row]))
        with pytest.raises(ValueError, match="row 1"):
            ev.load_labelled_fixture(path)
        with pytest.raises(ValueError, match="canonical_unit"):
            ev.load_labelled_fixture(path)


class TestStability:
    def test_option_order_flip_rate(self):
        out = ev.option_order_stability(["a", "b", "c", "d"], ["a", "b", "d", "c"])
        assert out["n"] == 4
        assert out["flips"] == 2
        assert out["flip_rate"] == 0.5

    def test_frozen_state_counts_changed_fields(self):
        records = [
            {"digest": "s1", "fields": {"choice": "a", "conf": 0.8}},
            {"digest": "s1", "fields": {"choice": "b", "conf": 0.6}},
            {"digest": "s2", "fields": {"choice": "a", "conf": 0.9}},
            {"digest": "s2", "fields": {"choice": "a", "conf": 0.9}},
        ]
        out = ev.frozen_state_variation(records)
        assert out["digests"] == 2
        assert out["repeated_states"] == 2
        assert out["fields"]["choice"]["states_changed"] == 1
        assert out["fields"]["conf"]["states_changed"] == 1
        assert out["fields"]["conf"]["min"] == 0.6
        assert out["fields"]["conf"]["max"] == 0.9


class TestCostLatency:
    def test_summary_totals_and_mean(self):
        out = ev.cost_latency_summary(
            [{"cost": 0.01, "latency_s": 2.0}, {"cost": 0.03, "latency_s": 4.0}]
        )
        assert out["n"] == 2
        assert out["total_cost"] == pytest.approx(0.04)
        assert out["mean_latency_s"] == pytest.approx(3.0)
        assert out["median_latency_s"] == pytest.approx(3.0)

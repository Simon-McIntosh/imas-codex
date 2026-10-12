"""Known-answer tests for the shared evaluation module.

Each metric is pinned against an answer known independently of it: a rank AUC
that fully separates the classes is 1.0, its reverse 0.0, a fully tied ranking
0.5, and a percentile interval brackets the estimate it was built from with
real width.  The operating point is checked to refuse the split it chose on.
"""

from __future__ import annotations

import json

import numpy as np
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


def _independent_difference_interval(
    labels, scores_a, scores_b, *, seed, n_resamples=1000, level=0.95
):
    """Percentile interval for AUC(a) - AUC(b) with each arm resampled alone.

    A reference computation, not the module's: every draw resamples the two
    arms on independent index sets, so it carries no pairing.  On correlated
    arms this is the wider interval; the paired interval the module builds must
    beat it.
    """
    y = np.asarray(labels, dtype=float)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = y.size
    draws: list[float] = []
    while len(draws) < n_resamples:
        idx_a = rng.integers(0, n, n)
        idx_b = rng.integers(0, n, n)
        if np.unique(y[idx_a]).size < 2 or np.unique(y[idx_b]).size < 2:
            continue
        draws.append(ev.roc_auc(y[idx_a], a[idx_a]) - ev.roc_auc(y[idx_b], b[idx_b]))
    tail = (1.0 - level) / 2.0
    lower, upper = np.percentile(draws, [tail * 100.0, (1.0 - tail) * 100.0])
    return {"lower": float(lower), "upper": float(upper)}


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
    def test_point_estimate_is_the_known_auc_and_interval_brackets_it(self):
        labels, scores = _separable_arm()
        # 12 of the 16 positive/negative pairs are ordered correctly, so the
        # rank AUC is 0.75 independently of any code under test.
        out = ev.bootstrap_interval(labels, scores, seed=7)
        assert out["point"] == 0.75
        assert out["lower"] < out["upper"]
        assert out["lower"] <= out["point"] <= out["upper"]
        assert out["seed"] == 7
        assert out["n_resamples"] == 1000


class TestBootstrapCopyGroups:
    """Near-duplicate copies resample together, not one copy at a time."""

    _BASE_LABELS = [1, 1, 1, 0, 0, 0]
    _BASE_SCORES = [0.9, 0.55, 0.4, 0.6, 0.35, 0.1]
    _COPIES = 4

    def _duplicated(self):
        labels = [lab for lab in self._BASE_LABELS for _ in range(self._COPIES)]
        scores = [score for score in self._BASE_SCORES for _ in range(self._COPIES)]
        groups = [
            item for item in range(len(self._BASE_LABELS)) for _ in range(self._COPIES)
        ]
        return labels, scores, groups

    def test_collapsing_copies_reproduces_the_deduplicated_interval(self):
        labels, scores, groups = self._duplicated()
        grouped = ev.bootstrap_interval(labels, scores, copy_groups=groups, seed=123)
        deduplicated = ev.bootstrap_interval(
            self._BASE_LABELS, self._BASE_SCORES, seed=123
        )
        assert grouped["lower"] == deduplicated["lower"]
        assert grouped["upper"] == deduplicated["upper"]

    def test_copy_group_interval_is_wider_than_per_observation(self):
        labels, scores, groups = self._duplicated()
        grouped = ev.bootstrap_interval(labels, scores, copy_groups=groups, seed=123)
        per_observation = ev.bootstrap_interval(labels, scores, seed=123)
        grouped_width = grouped["upper"] - grouped["lower"]
        per_observation_width = per_observation["upper"] - per_observation["lower"]
        assert grouped_width > per_observation_width


class TestPairedAuc:
    def test_known_aucs_and_their_difference(self):
        # Arm A separates both positives from both negatives (AUC 1.0); arm B
        # orders 3 of the 4 pairs correctly (AUC 0.75), so the difference is
        # 0.25 — all three read off the labels alone.
        labels = [1, 0, 1, 0]
        a = [0.9, 0.1, 0.8, 0.2]
        b = [0.9, 0.1, 0.2, 0.8]
        out = ev.paired_auc(labels, a, b)
        assert out["auc_a"] == 1.0
        assert out["auc_b"] == 0.75
        assert out["difference"] == 0.25


class TestPairedDifferenceInterval:
    """The interval over the paired difference, not over one arm alone."""

    _LABELS = [1, 1, 1, 1, 0, 0, 0, 0]
    _A = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]

    def test_reversed_arm_gives_unit_difference(self):
        out = ev.bootstrap_interval(
            self._LABELS, self._A, scores_b=self._A[::-1], seed=123
        )
        assert out["point"] == 1.0

    def test_identical_arms_give_zero_difference_bracketed_by_its_interval(self):
        out = ev.bootstrap_interval(self._LABELS, self._A, scores_b=self._A, seed=123)
        assert out["point"] == 0.0
        assert out["lower"] <= 0.0 <= out["upper"]

    def test_paired_interval_is_narrower_than_independent_resampling(self):
        rng = np.random.default_rng(5)
        n = 60
        labels = [1] * 30 + [0] * 30
        latent = rng.random(n)
        arm_a = np.array(labels) * 0.3 + 0.5 * (0.7 * latent + 0.3 * rng.random(n))
        arm_b = np.array(labels) * 0.3 + 0.5 * (0.7 * latent + 0.3 * rng.random(n))
        paired = ev.bootstrap_interval(labels, arm_a, scores_b=arm_b, seed=123)
        independent = _independent_difference_interval(labels, arm_a, arm_b, seed=123)
        paired_width = paired["upper"] - paired["lower"]
        independent_width = independent["upper"] - independent["lower"]
        assert paired_width < independent_width


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

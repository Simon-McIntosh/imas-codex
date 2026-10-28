"""Calibration-only rate matching and shared paired resampling."""

import json
from pathlib import Path

import numpy as np
import pytest

from imas_codex.discovery.base.evaluation import bootstrap_interval
from imas_codex.standard_names.review import cascade, cascade_calibration as cc


def test_matched_cutoff_and_mapped_score_cross_at_consumer_threshold():
    scores = [0.1, 0.2, 0.3, 0.4]
    fit = cc.matched_cutoff(scores, [0.1, 0.85, 0.7, 0.9])
    assert fit["cutoff"] == 0.3
    assert fit["pre_veto_accepts"] == fit["panel_accepts"] == 2
    cutoff = fit["cutoff"]
    below, at, above = [
        cascade.map_decision_score(s, cutoff)
        for s in (np.nextafter(cutoff, 0), cutoff, np.nextafter(cutoff, 1))
    ]
    assert below < 0.85 == at <= above
    assert [cascade.map_decision_score(s, cutoff) >= 0.85 for s in scores] == [
        False,
        False,
        True,
        True,
    ]


def test_matching_refuses_a_boundary_tie_instead_of_changing_the_target():
    with pytest.raises(ValueError, match="no inclusive cutoff matches 2/3"):
        cc.matched_cutoff([0, 0, 1], [0.9, 0.9, 0.1])


@pytest.mark.parametrize(
    "scores,panel",
    [
        ([], []),
        ([0.2], []),
        ([float("nan")], [1]),
        ([1.1], [1]),
        ([0.2], [float("nan")]),
    ],
)
def test_matching_refuses_unusable_scores(scores, panel):
    with pytest.raises(ValueError):
        cc.matched_cutoff(scores, panel)


def test_shared_bootstrap_accepts_statistics_and_keeps_copy_groups_paired():
    def statistic(y, scores):
        return float(np.mean(scores))

    result = bootstrap_interval(
        [1, 1],
        [1, 0],
        scores_b=[0.2, 0.0],
        copy_groups=["same", "same"],
        statistic=statistic,
        n_resamples=20,
        seed=17,
    )
    assert result["point"] == result["lower"] == result["upper"] == 0.4
    assert result["n_resamples"] == 20
    # The AUC default must still refuse a population with only one class.
    with pytest.raises(ValueError, match="both label classes"):
        bootstrap_interval([1, 1], [1, 0])


def test_undefined_precision_is_not_reported_as_zero():
    def statistic(y, scores):
        return cc._metric(y, scores, "accept_precision")

    missing = bootstrap_interval([1, 0], [0, 0], statistic=statistic, n_resamples=20)
    assert missing["point"] is None and missing["lower"] is None
    assert missing["n_resamples"] == 0
    control = bootstrap_interval([1, 0], [1, 0], statistic=statistic, n_resamples=20)
    assert control["point"] == 1 and control["n_resamples"] == 20


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not cc.DEFAULT_NAME_RECEIPTS.exists() or not cc.DEFAULT_DOCS_RECEIPTS.exists(),
    reason="frozen calibration receipts are not installed",
)
def test_matched_policy_rebuilds_from_frozen_scores_and_receipts():
    first = cc.build()
    assert first == cc.build()
    assert first == json.loads(cc.THRESHOLDS_PATH.read_text())
    assert first["name"]["score_kind"] == "logistic"
    assert first["docs"]["score_kind"] == "rubric"
    for axis in ("name", "docs"):
        block = first[axis]
        fit = block["matched_calibration"]
        report = block["report"]["calibration"]
        assert fit["pre_veto_accepts"] == fit["panel_accepts"]
        assert report["pre_veto"]["accept_rate"]["numerator"] == fit["panel_accepts"]
        assert (
            report["jev"]["accept_rate"]["numerator"]
            == fit["panel_accepts"] - report["veto"]["incremental_refine_count"]
        )
        assert set(report["route_counts"]) == {"accept", "refine"}
        assert "check_thresholds" not in block and "band_width" not in block["rubric"]
        for metric in cc.METRICS:
            assert metric in block["report"]["validation"]["paired_difference"]
    assert first["docs"]["report"]["calibration"]["veto"]["refine_count"] == 10


def test_a_missing_receipt_names_the_receipt_it_could_not_read():
    path = Path("/nonexistent/cascade-receipt.json")
    with pytest.raises(cc.ReceiptUnavailable) as excinfo:
        cc._read_json(path)
    assert str(path) in str(excinfo.value)

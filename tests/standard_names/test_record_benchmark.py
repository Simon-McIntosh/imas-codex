"""The record benchmark's tallies, paired comparison and control separation.

The benchmark is only a measure if its controls separate. These tests pin the
statistics that make a difference larger than its interval, the paired
comparison that refuses to call a small difference a win, and the two control
checks that refuse a benchmark in which the controls did not move.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from imas_codex.standard_names.advisory_benchmark import summarize_costs
from imas_codex.standard_names.benchmark import (
    BenchmarkControlDidNotSeparate,
    RecordBenchmarkArm,
    RecordBenchmarkRow,
    assert_controls_separate,
    assign_development_split,
    compare_record_arms,
    evaluate_controls,
    summarize_record_arm,
    wilson_interval,
)

ONE_PASS_LOCAL = RecordBenchmarkArm(
    name="one-pass/local", structure="one-pass", model="local/deepseek-v4.1-flash"
)
STAGED_LOCAL = RecordBenchmarkArm(
    name="staged/local", structure="staged", model="local/deepseek-v4.1-flash"
)
INCUMBENT = RecordBenchmarkArm(
    name="incumbent", structure="", model="", incumbent_unchanged=True
)
WITHHELD = RecordBenchmarkArm(
    name="withheld/local",
    structure="one-pass",
    model="local/deepseek-v4.1-flash",
    withheld_enclosing_text=True,
)


def rows_for(
    arm: str,
    verdicts: list[str],
    *,
    blockers: list[str | None] | None = None,
    changed: list[bool | None] | None = None,
    incumbent_right: list[bool | None] | None = None,
    prefix: str = "member",
) -> list[RecordBenchmarkRow]:
    return [
        RecordBenchmarkRow(
            member=f"{prefix}{index}",
            arm=arm,
            verdict=verdict,
            blocker=(blockers or [None] * len(verdicts))[index],
            changed_incumbent=(changed or [None] * len(verdicts))[index],
            incumbent_right=(incumbent_right or [None] * len(verdicts))[index],
            cost_usd=0.01,
            seconds=2.0,
            input_tokens=100,
            output_tokens=50,
        )
        for index, verdict in enumerate(verdicts)
    ]


def test_wilson_interval_keeps_width_at_the_extremes() -> None:
    low, high = wilson_interval(10, 10)

    assert 0.0 <= low < 1.0
    assert high <= 1.0
    assert low < high


def test_wilson_interval_bounds_and_inputs() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_arm_report_counts_blockers_drift_and_spend() -> None:
    rows = rows_for(
        "one-pass/local",
        ["right", "wrong", "blocked", "right"],
        blockers=[None, None, "grammar-gap", None],
        changed=[True, True, False, True],
        incumbent_right=[True, True, True, False],
    )

    report = summarize_record_arm(rows, ONE_PASS_LOCAL)

    assert report.total == 4
    assert report.right == 2
    assert report.wrong == 1
    assert report.blocked == 1
    assert report.blocked_by_type == {"grammar-gap": 1}
    assert report.share_right == 0.5
    assert report.changed_right_incumbent == 2
    assert report.right_incumbent_total == 3
    assert report.changed_right_incumbent_share == pytest.approx(2 / 3)
    assert report.cost_usd == pytest.approx(0.04)
    assert report.seconds_per_name == pytest.approx(2.0)
    assert report.tokens_read_per_name == pytest.approx(100.0)


def test_blockers_are_published_as_buckets_not_free_text() -> None:
    rows = rows_for(
        "one-pass/local",
        ["blocked", "blocked", "blocked", "blocked"],
        blockers=[
            "e",
            "missing_bound_source",
            "generation-failure:LLMStructuredCallError",
            "judge-failure:TimeoutError",
        ],
    )

    report = summarize_record_arm(rows, ONE_PASS_LOCAL)

    assert report.blocked_by_type == {
        "generation-failure": 1,
        "insufficient-source-text": 1,
        "judge-failure": 1,
        "other": 1,
    }


def test_paired_comparison_uses_the_shared_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_bootstrap(
        labels: object,
        scores: object,
        *,
        scores_b: object,
        statistic: object,
        n_resamples: int,
        seed: int,
    ) -> dict[str, object]:
        calls.update(
            labels=labels,
            scores=scores,
            scores_b=scores_b,
            statistic=statistic,
            n_resamples=n_resamples,
            seed=seed,
        )
        return {"lower": -0.2, "upper": 0.1, "n_resamples": n_resamples}

    monkeypatch.setattr(
        "imas_codex.discovery.base.evaluation.bootstrap_interval", fake_bootstrap
    )
    one_pass = rows_for("one-pass/local", ["right", "wrong", "right"])
    staged = rows_for("staged/local", ["right", "right", "wrong"])

    comparison = compare_record_arms(
        one_pass + staged, "one-pass/local", "staged/local", seed=7, draws=50
    )

    assert calls["scores"] == [1, 0, 1]
    assert calls["scores_b"] == [1, 1, 0]
    assert calls["n_resamples"] == 50
    assert calls["seed"] == 7
    statistic = calls["statistic"]
    assert callable(statistic)
    assert statistic([0.0], [1.0, 0.0, 1.0]) == pytest.approx(2 / 3)
    # The owner's interval is mean(a) - mean(b); the caller reports b - a.
    assert comparison["interval"] == pytest.approx([-0.1, 0.2])


def test_arm_report_refuses_a_member_scored_twice() -> None:
    rows = rows_for("one-pass/local", ["right", "right"])
    rows[1] = RecordBenchmarkRow(
        member=rows[0].member, arm=rows[1].arm, verdict="wrong"
    )

    with pytest.raises(ValueError, match="more than once"):
        summarize_record_arm(rows, ONE_PASS_LOCAL)


def test_blocked_row_must_name_a_type_and_only_a_blocked_row_carries_one() -> None:
    with pytest.raises(ValueError, match="blocker type"):
        RecordBenchmarkRow(member="m", arm="a", verdict="blocked")
    with pytest.raises(ValueError, match="only a blocked row"):
        RecordBenchmarkRow(member="m", arm="a", verdict="right", blocker="grammar-gap")


def test_paired_comparison_calls_a_clear_difference_a_win() -> None:
    one_pass = rows_for("one-pass/local", ["right"] * 10 + ["wrong"] * 10)
    staged = rows_for("staged/local", ["right"] * 18 + ["wrong"] * 2)
    rows = one_pass + staged

    comparison = compare_record_arms(rows, "one-pass/local", "staged/local", seed=7)

    assert comparison["members"] == 20
    assert comparison["right_a"] == 10
    assert comparison["right_b"] == 18
    assert comparison["only_b_right"] == 8
    assert comparison["only_a_right"] == 0
    assert comparison["difference"] == pytest.approx(0.4)
    assert comparison["interval"][0] > 0.0
    assert comparison["called_win"] is True


def test_paired_comparison_refuses_to_call_a_small_difference_a_win() -> None:
    one_pass = rows_for("one-pass/local", ["right", "wrong"] * 20)
    staged = rows_for("staged/local", ["right"] * 22 + ["wrong"] * 18)
    rows = one_pass + staged

    comparison = compare_record_arms(rows, "one-pass/local", "staged/local", seed=7)

    assert comparison["difference"] == pytest.approx(0.05)
    assert comparison["interval"][0] <= 0.0 <= comparison["interval"][1]
    assert comparison["called_win"] is False


def test_controls_separate_when_the_controls_move_the_right_way() -> None:
    members = [f"member{index}" for index in range(10)]
    rows = (
        rows_for("one-pass/local", ["right"] * 16 + ["wrong"] * 4)
        + rows_for("incumbent", ["right"] * 8 + ["wrong"] * 12)
        + rows_for("withheld/local", ["right"] * 1 + ["wrong"] * 9)
    )
    reports = [
        summarize_record_arm(rows, ONE_PASS_LOCAL),
        summarize_record_arm(rows, INCUMBENT),
        summarize_record_arm(rows, WITHHELD),
    ]

    controls = evaluate_controls(
        reports,
        rows,
        incumbent_arm="incumbent",
        withheld_arm="withheld/local",
        withheld_compared_to="one-pass/local",
        dependent_members=members,
        seed=7,
    )

    assert controls["incumbent"]["separated"] is True
    assert controls["withheld"]["separated"] is True
    assert controls["measuring"] is True
    assert_controls_separate(controls)


def test_withheld_control_at_full_arm_score_is_refused() -> None:
    members = [f"member{index}" for index in range(10)]
    same = ["right"] * 8 + ["wrong"] * 2
    rows = rows_for("one-pass/local", same) + rows_for("withheld/local", same)
    rows += rows_for("incumbent", ["wrong"] * 10)
    reports = [
        summarize_record_arm(rows, ONE_PASS_LOCAL),
        summarize_record_arm(rows, INCUMBENT),
        summarize_record_arm(rows, WITHHELD),
    ]

    controls = evaluate_controls(
        reports,
        rows,
        incumbent_arm="incumbent",
        withheld_arm="withheld/local",
        withheld_compared_to="one-pass/local",
        dependent_members=members,
        seed=7,
    )

    assert controls["withheld"]["separated"] is False
    assert controls["measuring"] is False
    with pytest.raises(BenchmarkControlDidNotSeparate, match="withheld-text control"):
        assert_controls_separate(controls)


def test_incumbent_control_at_repair_score_is_refused() -> None:
    rows = rows_for("one-pass/local", ["right"] * 10) + rows_for(
        "incumbent", ["right"] * 10
    )
    reports = [
        summarize_record_arm(rows, ONE_PASS_LOCAL),
        summarize_record_arm(rows, INCUMBENT),
    ]

    controls = evaluate_controls(reports, rows, incumbent_arm="incumbent")

    assert controls["incumbent"]["separated"] is False
    with pytest.raises(BenchmarkControlDidNotSeparate, match="incumbent-unchanged"):
        assert_controls_separate(controls)


def test_withheld_control_needs_the_members_its_text_supports() -> None:
    rows = rows_for("one-pass/local", ["right"] * 4) + rows_for(
        "withheld/local", ["wrong"] * 4
    )
    reports = [
        summarize_record_arm(rows, ONE_PASS_LOCAL),
        summarize_record_arm(rows, WITHHELD),
    ]

    with pytest.raises(ValueError, match="rests on"):
        evaluate_controls(
            reports,
            rows,
            withheld_arm="withheld/local",
            withheld_compared_to="one-pass/local",
        )


def test_development_split_is_deterministic_stratified_and_disjoint() -> None:
    members = [f"member{index}" for index in range(40)]
    strata = {
        member: "defective" if index % 3 == 0 else "sound"
        for index, member in enumerate(members)
    }

    first = assign_development_split(members, seed=11, strata=strata)
    second = assign_development_split(members, seed=11, strata=strata)

    assert first == second
    assert set(first) == set(members)
    assert set(first.values()) == {"development", "held_out"}
    assert len([value for value in first.values() if value == "development"]) == 20
    for label in ("defective", "sound"):
        sides = {first[member] for member in members if strata[member] == label}
        assert sides == {"development", "held_out"}


def test_development_split_refuses_duplicate_members() -> None:
    with pytest.raises(ValueError, match="unique"):
        assign_development_split(["a", "a"], seed=1)


def test_cost_summary_validates_both_sides_of_the_spend() -> None:
    with pytest.raises(ValueError, match="candidate cost"):
        summarize_costs(-0.01, 1.0, 25.0)
    summary = summarize_costs(20.0, 3.0, 25.0)
    assert summary.total_cost == pytest.approx(23.0)
    assert summary.within_authorized_ceiling is True


RUNNER = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/fragments/sn-case-file-generation/benchmark.py"
)


@pytest.fixture(scope="module")
def figure_runner():
    spec = importlib.util.spec_from_file_location("scfg_benchmark_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure_text_is_paths_and_no_figure_leaks(
    figure_runner, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No <text> element may reach the SVG, and no figure may stay managed.

    The published figure lost its labels under a renderer that could not
    resolve the declared font, so the text must be glyph paths. A figure that
    is cleared rather than closed stays registered in pyplot and accumulates
    in a long-lived process.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    document = {
        "final": {
            "arms": [
                {"arm": "incumbent", "share_right": 0.678, "interval": [0.574, 0.767]},
                {
                    "arm": "one-pass/frontier",
                    "share_right": 0.322,
                    "interval": [0.233, 0.426],
                },
            ]
        }
    }
    target = tmp_path / "heldout-arm-right.svg"
    monkeypatch.setattr(figure_runner, "FIGURE", target)

    figure_runner.render_figure(document)

    svg = target.read_text(encoding="utf-8")
    assert "<text" not in svg
    assert "font-family" not in svg
    assert "<path" in svg
    assert "#eeeeee" not in svg
    assert plt.get_fignums() == []

    figure_runner.render_figure(document)
    assert plt.get_fignums() == []

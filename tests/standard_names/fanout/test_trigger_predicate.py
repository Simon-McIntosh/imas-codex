"""Trigger arm assignment unit tests (refine fan-out A/B)."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.fanout.trigger import assign_arm


class TestAssignArm:
    def test_deterministic_for_same_inputs(self) -> None:
        a = assign_arm("electron_temperature", 1)
        b = assign_arm("electron_temperature", 1)
        assert a == b

    def test_arm_percent_zero_kills_all(self) -> None:
        for i in range(20):
            assert assign_arm(f"sn_{i}", i, arm_percent=0) == "off"

    def test_arm_percent_hundred_forces_all_on(self) -> None:
        for i in range(20):
            assert assign_arm(f"sn_{i}", i, arm_percent=100) == "on"

    def test_50_50_distribution_over_synthetic_cohort(self) -> None:
        on = 0
        off = 0
        for i in range(200):
            arm = assign_arm(f"electron_temperature_{i}", i % 3)
            if arm == "on":
                on += 1
            else:
                off += 1
        assert 70 <= on <= 130, f"on={on} not within 70..130"
        assert on + off == 200

    def test_different_cycle_indices_independent(self) -> None:
        arms = {assign_arm("test_sn", c) for c in range(10)}
        assert len(arms) == 2

    @pytest.mark.parametrize("percent", [25, 75])
    def test_percentage_skews_split(self, percent: int) -> None:
        on = sum(
            1
            for i in range(400)
            if assign_arm(f"sn_{i}", 0, arm_percent=percent) == "on"
        )
        if percent == 25:
            assert 60 <= on <= 160
        else:
            assert 240 <= on <= 340
